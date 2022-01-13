"""The set of functions to perform Hebbian Learning."""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from PIL import ImageEnhance as imEnh


class GenerateArrayFromImage:
    """
    Generate both a binary array out of an image and its flawed version.

    Parameters:
    - image: the image to create the array out.
    - contrast_enhance: an int that controls the sharpeness for the final bitmap.
    - flaw_intensity: a float between 0 and 1 that controls how distorted will be the flawed image.
    - plot: bool that controls whether to show the outcome of the process.

    https://linux.ime.usp.br/~robotenique//computer%20science/python/2017/09/17/image-grid-python.html
    """

    def __init__(self, image, contrast_enhance=10, flaw_intensity=0.1, plot=False):
        msg = "Intensity should be between 0 and 1"
        assert flaw_intensity >= 0 and flaw_intensity <= 1, msg
        self.image = image
        self.contrast_enhance = contrast_enhance
        self.flaw_intensity = flaw_intensity
        self.plot = plot

    def run(self):
        """Generate both the image and its flawed version."""
        self.image_array = self._get_array_from_image()
        self.flaw_array = self._flaw_bitmap()
        if self.plot:
            self.plot_bitmap(self.image_array)
            self.plot_bitmap(self.flaw_array)

    @staticmethod
    def plot_bitmap(array):
        """Plot the given array."""
        plt.gray()
        plt.matshow(array)
        plt.show()

    def _get_array_from_image(self):
        """Output a np.array with binary values out of the selected image."""
        img = im.open(self.image)

        # Resizing
        sz = (40, 40)
        img.thumbnail(sz, im.ANTIALIAS)
        sz = (img.size[0], img.size[1])
        img = imEnh.Contrast(img).enhance(self.contrast_enhance).convert("1")

        # Create a white background and paste the image onto it
        bkg = im.new("RGBA", sz, (255, 255, 255, 0))
        bkg.paste(img, (0, 0))

        # Create the array out of the image
        tarr = np.asarray(bkg.convert("1"))
        base = np.full_like(tarr, 1, dtype=np.int)
        base[tarr] = 0
        return base

    def _flaw_bitmap(self):
        """Swap 0s by 1s and 1s by 0s randomly."""
        flawed = self.image_array.copy()
        intensity_complement = 1 - self.flaw_intensity
        flawed[flawed == 0] = np.random.binomial(
            1, p=self.flaw_intensity, size=(flawed == 0).sum()
        )
        flawed[flawed == 1] = np.random.binomial(
            1, p=intensity_complement, size=(flawed == 1).sum()
        )
        return flawed


class Hebbian:
    """Perform Hebbian learning over some image."""

    def __init__(self, image_array, flawed_array, bias=None, max_iterations=10):
        """
        Class constructor.

        Parameters:
            * image_array: a one dimensional array with binary values [-1, 1] representing pixels
            on an image or a two dimensional array representing several images.
            * flawed_array: a distorted image of the initial image or one of them.
            * bias: an array representing the bias for each neuron.
            * max_iterations: the max number of iterations to arrive to convergence.
        """
        self.image_array = image_array
        self.history = np.array([flawed_array])  # Store iterations in a 2 x n array
        self.bias = bias
        self.max_iterations = max_iterations

    @staticmethod
    def _get_rid_of_the_diagonal(cartesian_product):
        """
        Given a nxn array remove the elements in the main diagonal and flatten the array.

        When calculating weights and net inputs we also get operations of each element with
        itself. This function gets rid of them.

        Parameters:
            * A n x n square array.
        Returns:
            * A n x n-1 array
        """
        n = cartesian_product.shape[0]
        assert n == cartesian_product.shape[1]  # ensure the array is square
        diag = np.diag_indices(n)
        cartesian_product[diag] = np.nan
        cartesian_product = cartesian_product.ravel()
        cartesian_product = cartesian_product[~np.isnan(cartesian_product)]

        # Transform the array into n x n-1
        final_size = (n, n - 1)
        return cartesian_product.reshape(final_size)

    @staticmethod
    def _sign(activation_value):
        """
        Calculate the output of a neuron.

        We activate all the neurons by default and then switch off those ones that have an
        activation value less than 0.
        """
        signed_array = np.ones(activation_value.shape)
        signed_array[activation_value < 0] = -1
        return signed_array

    def run(self):
        weights = self._get_total_weight()
        while self.max_iterations:
            net_input = self._get_net_input(weights, self.history[-1])
            activation_value = self._get_activation_value(net_input)
            signed_array = self._sign(activation_value)
            self.history = np.append(self.history, [signed_array], axis=0)
            self.max_iterations -= 1
            if self.has_converged() or self._row_already_exists():
                return

    def _get_total_weight(self):
        """Get the overall weight depending the number of stored images."""
        single_image = len(self.image_array.shape) == 1
        if single_image:
            return 1 / self.image_array.size * self._get_weights(self.image_array)

        no_of_neurons = self.image_array.shape[1]
        total_weight = np.zeros(shape=(no_of_neurons, no_of_neurons))
        for image_array in self.image_array:
            weighted_image = self._get_weights(image_array)
            total_weight += weighted_image
        return 1 / self.image_array.shape[1] * total_weight

    @staticmethod
    def _get_weights(image_array):
        """
        Get the weights for all the possible pairs of elements in an image.

        The idea behind the weigths is that they keep the original memory. They may well
        represent the role of myelin favoring or banning the transmission of information among
        neurons. Threrefore, these weights are set at the beginning with the original memory
        and then are used over iterations to drive partial informations to the memories
        stored. They are created by calculating the average product between one neuron and all
        the ones it's connected to.

        Note that this function will return an n x n array where n is the size of the
        original. In the example above this means that a neuron has a weight set with itself.
        This is a convenience as afterwards we will have to multiply the weight value and the
        input comming from the neigbour neuron and once that is done we remove the diagonals.
        """
        # create an exteded version of the array such that we can perform a matrix
        # multiplication
        size = image_array.size
        extended = np.full(shape=(size, size), fill_value=image_array)

        # perform the multiplication
        return image_array * extended.T

    def _get_net_input(self, weights, neuron_activation):
        """
        Calculate the net input for all the neurons in the array.

        Parameters:
            * weights: a nxn array with the weights of each neuron with itself and the others.
            * neuron_activation: a n-size vector with the activation value for each neuron. This
            represents the flawed version of the array in the beginning or the intermediate
            activation values while iterating.
        Returns:
            A n-size vector.
        """
        # Compute the net input
        raw_net_input = neuron_activation * weights.T

        # Getting rid of the diagonal means that the array becomes n x n-1
        raw_net_input = self._get_rid_of_the_diagonal(raw_net_input)
        return raw_net_input.sum(axis=1)

    def _get_activation_value(self, net_input):
        """
        Calculate total activation value for a neuron.

        The total activation value for a neuron is the net input minus the bias.

        Parameters:
            * net_input: a n-size vector representing the net input for each neuron.
            * bias: a n-size vector representing the bias for each neuron.
        Returns:
            * a n-size vector.
        """
        activation_value = net_input.copy()
        if self.bias is not None:
            assert activation_value.shape == self.bias.shape
            activation_value -= self.bias
        return activation_value

    def has_converged(self):
        """Determine whether the flawed image converged to the original one."""
        return (self.image_array == self.history[-1]).all()

    def looped_back_to_initial(self):
        """Determine whether some iteration returned to the original flawed image."""
        return self.history.shape[0] > 1 and (self.history[0] == self.history[-1]).all()

    def _row_already_exists(self):
        """Determine whether the same row already appeared before."""
        if self.history.shape[1] > 1:
            for row in self.history[:-1]:
                if (row == self.history[-1]).all():
                    return True
