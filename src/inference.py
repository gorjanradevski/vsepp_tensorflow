import tensorflow as tf
import argparse
import logging
from tqdm import tqdm
import os
import absl.logging

from datasets import FlickrDataset
from evaluators import Evaluator
from models import VsePpModel
from loaders import InferenceLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

# https://github.com/abseil/abseil-py/issues/99
absl.logging.set_verbosity("info")
absl.logging.set_stderrthreshold("info")


def inference(
    images_path: str,
    texts_path: str,
    test_imgs_file_path,
    batch_size: int,
    prefetch_size: int,
    checkpoint_path: str,
    joint_space: int,
    num_layers: int,
) -> None:
    """Performs inference on the Flickr8k/30k test set.

    Args:
        images_path: A path where all the images are located.
        texts_path: Path where the text doc with the descriptions is.
        test_imgs_file_path: Path to a file with the test image names.
        batch_size: The batch size to be used.
        prefetch_size: How many batches to prefetch.
        checkpoint_path: Path to a valid model checkpoint.
        joint_space: The size of the joint latent space.
        num_layers: The number of rnn layers.

    Returns:
        None

    """
    dataset = FlickrDataset(images_path, texts_path)
    # Getting the vocabulary size of the train dataset
    test_image_paths, test_captions = dataset.get_data(test_imgs_file_path)
    logger.info("Test dataset created...")
    evaluator_test = Evaluator(len(test_image_paths), joint_space)

    logger.info("Test evaluator created...")

    # Resetting the default graph and setting the random seed
    tf.reset_default_graph()

    loader = InferenceLoader(test_image_paths, test_captions, batch_size, prefetch_size)
    images, captions, captions_lengths = loader.get_next()
    logger.info("Loader created...")

    model = VsePpModel(images, captions, captions_lengths, joint_space, num_layers)
    logger.info("Model created...")
    logger.info("Inference is starting...")

    with tf.Session() as sess:

        # Initializers
        model.init(sess, checkpoint_path)
        try:
            with tqdm(total=len(test_image_paths)) as pbar:
                while True:
                    loss, lengths, embedded_images, embedded_captions = sess.run(
                        [
                            model.loss,
                            model.captions_len,
                            model.image_encoded,
                            model.text_encoded,
                        ]
                    )
                    evaluator_test.update_metrics(loss)
                    evaluator_test.update_embeddings(embedded_images, embedded_captions)
                    pbar.update(len(lengths))
        except tf.errors.OutOfRangeError:
            pass

            logger.info(
                f"The image2text recall at (1, 5, 10) is: "
                f"{evaluator_test.image2text_recall_at_k()}"
            )

            logger.info(
                f"The text2image recall at (1, 5, 10) is: "
                f"{evaluator_test.text2image_recall_at_k()}"
            )


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
        args.images_path,
        args.texts_path,
        args.test_imgs_file_path,
        args.batch_size,
        args.prefetch_size,
        args.checkpoint_path,
        args.joint_space,
        args.num_layers,
    )


def parse_args():
    """Parse command line arguments.

    Returns:
        Arguments

    """
    parser = argparse.ArgumentParser(
        "Performs inference on the Flickr8k and Flickr30k datasets."
        "Defaults to the Flickr8k dataset."
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
        "--test_imgs_file_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr_8k.devImages.txt",
        help="Path to the file where the test images names are included.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to a model checkpoint."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch."
    )
    parser.add_argument(
        "--prefetch_size", type=int, default=5, help="The size of prefetch on gpu."
    )
    parser.add_argument(
        "--joint_space", type=int, default=5, help="The size of the joint space."
    )
    parser.add_argument(
        "--num_layers", type=int, default=5, help="The number of rnn layers."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
