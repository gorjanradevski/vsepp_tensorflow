import sys
import logging
import numpy as np

from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, num_samples: int = 0, num_features: int = 0):
        self.loss = 0.0
        self.best_loss = sys.maxsize
        self.best_image2text_recall_at_k = (-1.0, -1.0, -1.0)
        self.cur_image2text_recall_at_k = (-1.0, -1.0, -1.0)
        self.best_text2image_recall_at_k = (-1.0, -1.0, -1.0)
        self.cur_text2image_recall_at_k = (-1.0, -1.0, -1.0)
        self.index_update = 0
        self.num_samples = num_samples
        self.num_features = num_features
        self.embedded_images = np.zeros((self.num_samples, self.num_features))
        self.embedded_captions = np.zeros((self.num_samples, self.num_features))

    def reset_all_vars(self) -> None:
        self.loss = 0
        self.index_update = 0
        self.embedded_images = np.zeros((self.num_samples, self.num_features))
        self.embedded_captions = np.zeros((self.num_samples, self.num_features))
        self.cur_text2image_recall_at_k = -1.0
        self.cur_image2text_recall_at_k = -1.0

    def update_metrics(self, loss: float) -> None:
        self.loss += loss

    def update_embeddings(
        self, embedded_images: np.ndarray, embedded_captions: np.ndarray
    ) -> None:
        num_samples = embedded_images.shape[0]
        self.embedded_images[
            self.index_update : self.index_update + num_samples, :
        ] = embedded_images
        self.embedded_captions[
            self.index_update : self.index_update + num_samples, :
        ] = embedded_captions
        self.index_update += num_samples

    def is_best_loss(self) -> bool:
        if self.loss < self.best_loss:
            return True
        return False

    def update_best_loss(self):
        self.best_loss = self.loss

    def is_best_image2text_recall_at_k(self) -> bool:
        self.cur_image2text_recall_at_k = self.image2text_recall_at_k()
        if sum(self.cur_image2text_recall_at_k) > sum(self.best_image2text_recall_at_k):
            return True
        return False

    def update_best_image2text_recall_at_k(self):
        self.best_image2text_recall_at_k = self.cur_image2text_recall_at_k

    def is_best_text2image_recall_at_k(self) -> bool:
        self.cur_text2image_recall_at_k = self.text2image_recall_at_k()
        if sum(self.cur_text2image_recall_at_k) > sum(self.best_text2image_recall_at_k):
            return True
        return False

    def update_best_text2image_recall_at_k(self):
        self.best_text2image_recall_at_k = self.cur_text2image_recall_at_k

    def image2text_recall_at_k(self) -> Tuple[float, float, float]:
        """Computes the recall at K when doing image to text retrieval and updates the
        object variable.

        Returns:
            The recall at 1, 5, 10.

        """
        num_images = self.embedded_images.shape[0] // 5
        ranks = np.zeros(num_images)
        for index in range(num_images):
            # Get query image
            query_image = self.embedded_images[5 * index]
            # Similarities
            similarities = np.dot(query_image, self.embedded_captions.T).flatten()
            indices = np.argsort(similarities)[::-1]
            # Score
            rank = sys.maxsize
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(indices == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        return r1, r5, r10

    def text2image_recall_at_k(self) -> Tuple[float, float, float]:
        """Computes the recall at K when doing text to image retrieval and updates the
        object variable.

        Returns:
            The recall at 1, 5, 10.

        """
        num_captions = self.embedded_captions.shape[0]
        ranks = np.zeros(num_captions)
        for index in range(num_captions):
            # Get query captions
            query_captions = self.embedded_captions[5 * index : 5 * index + 5]
            # Similarities
            similarities = np.dot(query_captions, self.embedded_images[0::5].T)
            inds = np.zeros(similarities.shape)
            for i in range(len(inds)):
                inds[i] = np.argsort(similarities[i])[::-1]
                ranks[5 * index + i] = np.where(inds[i] == index)[0][0]

        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        return r1, r5, r10
