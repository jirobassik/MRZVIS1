import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from matrix_oper import transpose_matrix, multiplication_matrix, number_on_matrix


class NeuralCompress:
    def __init__(self, input_img):
        self.img_height = 256
        self.img_width = 256

        self.img_array = numpy.array(self.img_to_array(input_img))

        self.block_height = 4
        self.block_width = 4
        self.alpha = 0.001

        self.number_of_blocks = int((self.img_height * self.img_width) / (self.block_height * self.block_width))

        self.input_layer_size = self.block_height * self.block_height * 3
        self.hidden_layer_size = 16

        self.blocks = self.__to_blocks().reshape((self.number_of_blocks, 1, self.input_layer_size))

        self.first_layer = numpy.random.rand(self.input_layer_size, self.hidden_layer_size) * 2 - 1

        temp_layer = numpy.copy(self.first_layer)
        self.second_layer = numpy.array(transpose_matrix(temp_layer))

        self.error_max = 2000.0

    def __get_block(self, i, j):
        block = []
        for y in range(self.block_height):
            for x in range(self.block_width):
                for color in range(3):
                    block.append(self.img_array[i * self.block_height + y,
                                                j * self.block_width + x, color])
        return block

    def __to_blocks(self):
        blocks = []
        for i in range(self.img_height // self.block_height):
            for j in range(self.img_width // self.block_width):
                block = self.__get_block(i, j)
                blocks.append(block)
        return numpy.array(blocks)

    def __to_array(self, blocks):
        array = []
        blocks_in_line = self.img_width // self.block_width
        for i in range(self.img_height // self.block_height):
            for y in range(self.block_height):
                line = []
                for j in range(blocks_in_line):
                    for x in range(self.block_width):
                        pixel = []
                        for color in range(3):
                            pixel.append(
                                blocks[i * blocks_in_line + j, (y * self.block_width * 3) + (x * 3) + color])
                        line.append(pixel)
                array.append(line)
        return numpy.array(array)

    @staticmethod
    def __display_array(img_array):
        img_array = 1 * (img_array + 1) / 2
        plt.imshow(img_array)
        plt.show()

    @staticmethod
    def img_to_array(img_):
        return (2.0 * img_ / 1.0) - 1.0

    @staticmethod
    def normalize_matrix(matrix):
        for i_f in range(len(matrix[0])):
            s = 0
            for j_f in range(len(matrix)):
                s += matrix[j_f][i_f] * matrix[j_f][i_f]
            s = math.sqrt(s)
            for j_f in range(len(matrix)):
                matrix[j_f][i_f] = matrix[j_f][i_f] / s

    def display_array(self):
        self.img_array = 1 * (self.img_array + 1) / 2
        plt.imshow(self.img_array)
        plt.show()

    def train(self):
        error_max = self.error_max
        error_current = error_max + 1
        epoch = 0

        while error_current > error_max:
            error_current = 0
            epoch += 1
            for i in self.blocks:
                y = numpy.array(multiplication_matrix(i, self.first_layer))
                x1 = numpy.array(multiplication_matrix(y, self.second_layer))
                dx = x1 - i
                mult_matrix = multiplication_matrix(transpose_matrix(i), dx)
                mult_mult_matrix = numpy.array(multiplication_matrix(numpy.array(mult_matrix),
                                                                     transpose_matrix(self.second_layer)))
                self.first_layer -= numpy.array(number_on_matrix(self.alpha, mult_mult_matrix))
                self.normalize_matrix(self.first_layer)
                self.second_layer -= numpy.array(
                    number_on_matrix(self.alpha, multiplication_matrix(transpose_matrix(y), dx)))
                self.normalize_matrix(self.second_layer)

            for i in self.blocks:
                dx = numpy.array(
                    multiplication_matrix(multiplication_matrix(i, self.first_layer), self.second_layer) - i)
                error = (dx * dx).sum()
                error_current += error

            print('Epoch:', epoch)
            print('Error:', error_current)

        print('Z:', self.calc_compression_rate())

    def calc_compression_rate(self):
        return ((numpy.size(self.blocks) * self.number_of_blocks) / (
                (numpy.size(self.blocks) + self.number_of_blocks) * self.hidden_layer_size + 2)) / 100

    def restore_img(self):
        result = []
        for block in self.blocks:
            result.append(
                numpy.array(multiplication_matrix(multiplication_matrix(block, self.first_layer), self.second_layer)))
        result = numpy.array(result)
        return self.__to_array(result.reshape(
            self.number_of_blocks, self.input_layer_size))

    def show_result(self):
        self.__display_array(self.restore_img())


img = mpimg.imread('cat.png')

au = NeuralCompress(img)
au.display_array()

au.train()
au.show_result()
