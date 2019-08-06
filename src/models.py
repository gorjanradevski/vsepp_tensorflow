import tensorflow as tf
import logging
import tensorflow_hub as hub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VsePpModel:
    def __init__(
        self,
        images: tf.Tensor,
        captions: tf.Tensor,
        captions_len: tf.Tensor,
        joint_space: int,
        num_layers: int,
    ):
        # Get images, captions, lengths and labels
        self.images = images
        self.captions = captions
        self.captions_len = captions_len
        # Create placeholders
        self.learning_rate = tf.placeholder_with_default(0.0, None, name="lr")
        self.margin = tf.placeholder_with_default(0.0, None, name="margin")
        self.weight_decay = tf.placeholder_with_default(0.0, None, name="weight_decay")
        self.clip_value = tf.placeholder_with_default(0.0, None, name="clip_value")
        self.decay_steps = tf.placeholder_with_default(0.0, None, name="decay_steps")
        # Build model
        self.image_encoded = self.image_encoder_graph(joint_space)
        logger.info("Image encoder graph created...")
        self.text_encoded = self.text_encoder_graph(joint_space, num_layers)
        logger.info("Text encoder graph created...")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.loss = self.triplet_loss(self.margin)
        self.optimize = self.apply_gradients_op(
            self.loss, self.learning_rate, self.clip_value, self.decay_steps
        )
        self.saver_loader = tf.train.Saver()
        logger.info("Graph creation finished...")

    def image_encoder_graph(self, joint_space: int) -> tf.Tensor:
        """Extract higher level features from the image using a resnet152 pretrained on
        ImageNet.

        Args:
            joint_space: The space where the encoded images and text are going to be
            projected to.

        Returns:
            The encoded image.

        """
        with tf.variable_scope("image_encoder"):
            resnet = hub.Module(
                "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/3"
            )
            features = resnet(self.images, signature="image_feature_vector")
            linear_layer = tf.layers.dense(
                features,
                joint_space,
                kernel_initializer=tf.glorot_uniform_initializer(),
            )

            return linear_layer

    def text_encoder_graph(self, joint_space: int, num_layers: int):
        """Encodes the text it gets as input using a bidirectional rnn.

        Args:
            joint_space: The space where the encoded images and text are going to be
            projected to.
            num_layers: The number of layers in the Bi-RNN.

        Returns:
            The encoded text.

        """
        with tf.variable_scope(name_or_scope="text_encoder"):
            elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
            embeddings = elmo(
                inputs={"tokens": self.captions, "sequence_len": self.captions_len},
                signature="tokens",
                as_dict=True,
            )["elmo"]
            cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.GRUCell(joint_space) for _ in range(num_layers)]
            )
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.GRUCell(joint_space) for _ in range(num_layers)]
            )
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                embeddings,
                sequence_length=self.captions_len,
                dtype=tf.float32,
            )
            last_fw = self.last_relevant(output_fw, self.captions_len)
            last_bw = self.last_relevant(output_bw, self.captions_len)

            return tf.add(last_fw, last_bw) / 2

    @staticmethod
    def last_relevant(output: tf.Tensor, length: tf.Tensor) -> tf.Tensor:
        """Returns the last relevant state from the RNN.

        # https://danijar.com/variable-sequence-lengths-in-tensorflow/

        Args:
            output: All hidden states
            length: The length of the hidden states.

        Returns:
            The last relevant hidden state.

        """
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)

        return relevant

    def triplet_loss(self, margin: float) -> tf.Tensor:
        """Computes the final loss of the model.

        1. Computes the Triplet loss: https://arxiv.org/abs/1707.05612 (Batch hard)
        2. Computes the L2 loss.
        3. Adds all together to compute the loss.

        Args:
            margin: The contrastive margin.

        Returns:
            The final loss to be optimized.

        """
        with tf.variable_scope(name_or_scope="loss"):
            scores = tf.matmul(self.image_encoded, self.text_encoded, transpose_b=True)

            diagonal = tf.diag_part(scores)
            # Compare every diagonal score to scores in its column
            cost_s = tf.maximum(0.0, margin - tf.reshape(diagonal, [-1, 1]) + scores)
            # Compare every diagonal score to scores in its row
            cost_im = tf.maximum(0.0, margin - diagonal + scores)

            # Clear diagonals
            cost_s = tf.linalg.set_diag(cost_s, tf.zeros(tf.shape(cost_s)[0]))
            cost_im = tf.linalg.set_diag(cost_im, tf.zeros(tf.shape(cost_im)[0]))

            # For each positive pair (i,s) pick the hardest contrastive image
            cost_s = tf.reduce_max(cost_s, axis=1)
            # For each positive pair (i,s) pick the hardest contrastive sentence
            cost_im = tf.reduce_max(cost_im, axis=0)

            triplet_loss = tf.reduce_sum(cost_s) + tf.reduce_sum(cost_im)

            l2_loss = (
                tf.add_n(
                    [
                        tf.nn.l2_loss(v)
                        for v in tf.trainable_variables()
                        if "bias" not in v.name
                    ]
                )
                * self.weight_decay
            )

            return triplet_loss + l2_loss

    def apply_gradients_op(
        self, loss: tf.Tensor, learning_rate: float, clip_value: int, decay_steps: float
    ) -> tf.Operation:
        """Applies the gradients on the variables.

        Args:
            loss: The computed loss.
            learning_rate: The optimizer learning rate.
            clip_value: The clipping value.
            decay_steps: Decay the learning rate every decay_steps.

        Returns:
            An operation node to be executed in order to apply the computed gradients.

        """
        with tf.variable_scope(name_or_scope="optimizer"):
            learning_rate = tf.train.exponential_decay(
                learning_rate,
                self.global_step,
                decay_steps,
                0.5,
                staircase=True,
                name="lr_decay",
            )
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)

            return optimizer.apply_gradients(
                zip(gradients, variables), global_step=self.global_step
            )

    def init(self, sess: tf.Session, checkpoint_path: str = None) -> None:
        """Initializes all variables in the graph.

        Args:
            sess: The active session.
            checkpoint_path: Path to a valid checkpoint.

        Returns:
            None

        """
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        if checkpoint_path is not None:
            self.saver_loader.restore(sess, checkpoint_path)

    def save_model(self, sess: tf.Session, save_path: str) -> None:
        """Dumps the model definition.

        Args:
            sess: The active session.
            save_path: Where to save the model.

        Returns:

        """
        self.saver_loader.save(sess, save_path)
