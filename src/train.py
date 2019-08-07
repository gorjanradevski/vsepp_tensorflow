import tensorflow as tf
import argparse
import logging
from tqdm import tqdm
import os
import absl.logging

from datasets import FlickrDataset
from loaders import TrainValLoader
from models import VsePpModel
from evaluators import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

# https://github.com/abseil/abseil-py/issues/99
absl.logging.set_verbosity("info")
absl.logging.set_stderrthreshold("info")


def train(
    images_path: str,
    texts_path: str,
    train_imgs_file_path: str,
    val_imgs_file_path: str,
    joint_space: int,
    num_layers: int,
    learning_rate: float,
    margin: float,
    clip_val: float,
    decay_rate: int,
    weight_decay: float,
    batch_size: int,
    prefetch_size: int,
    epochs: int,
    save_model_path: str,
) -> None:
    """Starts a training session with the Flickr8k dataset.

    Args:
        images_path: A path where all the images are located.
        texts_path: Path where the text doc with the descriptions is.
        train_imgs_file_path: Path to a file with the train image names.
        val_imgs_file_path: Path to a file with the val image names.
        joint_space: The space where the encoded images and text will be projected.
        num_layers: Number of layers of the rnn.
        epochs: The number of epochs to train the model.
        batch_size: The batch size to be used.
        prefetch_size: How many batches to keep on GPU ready for processing.
        save_model_path: Where to save the model.
        learning_rate: The learning rate.
        weight_decay: The L2 loss constant.
        margin: The contrastive margin.
        clip_val: The max grad norm.
        decay_rate: When to decay the learning rate.

    Returns:
        None

    """
    dataset = FlickrDataset(images_path, texts_path)
    train_image_paths, train_captions = dataset.get_data(train_imgs_file_path)
    val_image_paths, val_captions = dataset.get_data(val_imgs_file_path)
    logger.info("Dataset created...")
    evaluator_val = Evaluator(len(val_image_paths), joint_space)
    logger.info("Evaluators created...")

    # Resetting the default graph
    tf.reset_default_graph()
    loader = TrainValLoader(
        train_image_paths,
        train_captions,
        val_image_paths,
        val_captions,
        batch_size,
        prefetch_size,
    )
    images, captions, captions_lengths = loader.get_next()
    logger.info("Loader created...")

    decay_steps = decay_rate * len(train_image_paths) / batch_size
    model = VsePpModel(images, captions, captions_lengths, joint_space, num_layers)
    logger.info("Model created...")
    logger.info("Training is starting...")

    with tf.Session() as sess:
        # Initializers
        model.init(sess)
        for e in range(epochs):
            # Reset evaluators
            evaluator_val.reset_all_vars()

            # Initialize iterator with train data
            sess.run(loader.train_init)
            try:
                with tqdm(total=len(train_image_paths)) as pbar:
                    while True:
                        _, loss, lengths = sess.run(
                            [model.optimize, model.loss, model.captions],
                            feed_dict={
                                model.weight_decay: weight_decay,
                                model.learning_rate: learning_rate,
                                model.margin: margin,
                                model.decay_steps: decay_steps,
                                model.clip_value: clip_val,
                            },
                        )
                        pbar.update(len(lengths))
                        pbar.set_postfix({"Batch loss": loss})
            except tf.errors.OutOfRangeError:
                pass

            # Initialize iterator with validation data
            sess.run(loader.val_init)
            try:
                with tqdm(total=len(val_image_paths)) as pbar:
                    while True:
                        loss, lengths, embedded_images, embedded_captions = sess.run(
                            [
                                model.loss,
                                model.captions,
                                model.image_encoded,
                                model.text_encoded,
                            ]
                        )
                        evaluator_val.update_metrics(loss)
                        evaluator_val.update_embeddings(
                            embedded_images, embedded_captions
                        )
                        pbar.update(len(lengths))
            except tf.errors.OutOfRangeError:
                pass

            if evaluator_val.is_best_recall_at_k():
                evaluator_val.update_best_recall_at_k()
                logger.info("=============================")
                logger.info(
                    f"Found new best on epoch {e+1}!! Saving model!\n"
                    f"Current image-text recall at 1, 5, 10: "
                    f"{evaluator_val.best_image2text_recall_at_k} \n"
                    f"Current text-image recall at 1, 5, 10: "
                    f"{evaluator_val.best_text2image_recall_at_k}"
                )
                logger.info("=============================")
                model.save_model(sess, save_model_path)


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    train(
        args.images_path,
        args.texts_path,
        args.train_imgs_file_path,
        args.val_imgs_file_path,
        args.joint_space,
        args.num_layers,
        args.learning_rate,
        args.margin,
        args.clip_val,
        args.decay_rate,
        args.weight_decay,
        args.batch_size,
        args.prefetch_size,
        args.epochs,
        args.save_model_path,
    )


def parse_args():
    """Parse command line arguments.

    Returns:
        Arguments

    """
    parser = argparse.ArgumentParser(
        description="Performs multi_hop_attention on the Flickr8k and Flicrk30k"
        "dataset. Defaults to the Flickr8k dataset."
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_Dataset",
        help="Path where all images are.",
    )
    parser.add_argument(
        "--texts_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr8k.token.txt",
        help="Path to the file where the image to caption mappings are.",
    )
    parser.add_argument(
        "--train_imgs_file_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr_8k.trainImages.txt",
        help="Path to the file where the train images names are included.",
    )
    parser.add_argument(
        "--val_imgs_file_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr_8k.devImages.txt",
        help="Path to the file where the validation images names are included.",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/untitled",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of epochs to train the model excluding the vgg.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="The size of the batch."
    )
    parser.add_argument(
        "--prefetch_size", type=int, default=5, help="The size of prefetch on gpu."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="The learning rate."
    )
    parser.add_argument(
        "--joint_space",
        type=int,
        default=1024,
        help="The joint space where the encodings will be projected.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="The joint space where the encodings will be projected.",
    )
    parser.add_argument(
        "--margin", type=float, default=0.2, help="The contrastive margin."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="The L2 constant."
    )
    parser.add_argument(
        "--clip_val", type=float, default=2.0, help="The clipping threshold."
    )
    parser.add_argument(
        "--decay_rate", type=int, default=4, help="When to decay the learning rate."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
