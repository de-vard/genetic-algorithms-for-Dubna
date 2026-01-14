from PIL import Image, ImageDraw
import numpy as np
from skimage.metrics import structural_similarity
import cv2
import matplotlib.pyplot as plt


class ImageTest:
    def __init__(self, imagePath, polygonSize):
        """
        Инициализация объекта:
        - загружает эталонное изображение
        - сохраняет количество вершин многоугольника
        """
        self.refImage = Image.open(imagePath)
        self.polygonSize = polygonSize

        # Размеры изображения
        self.width, self.height = self.refImage.size
        self.numPixels = self.width * self.height

        # Эталонное изображение в формате OpenCV
        self.refImageCv2 = self.toCv2(self.refImage)

    def polygonDataToImage(self, polygonData):
        """
        Создаёт изображение из набора многоугольников
        polygonData — список параметров всех многоугольников
        """

        # Создаём пустое изображение
        image = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(image, 'RGBA')

        # Размер блока данных одного многоугольника
        chunkSize = self.polygonSize * 2 + 4
        polygons = self.list2Chunks(polygonData, chunkSize)

        # Рисуем каждый многоугольник
        for poly in polygons:
            index = 0
            vertices = []

            # Получаем координаты вершин
            for _ in range(self.polygonSize):
                x = int(poly[index] * self.width)
                y = int(poly[index + 1] * self.height)
                vertices.append((x, y))
                index += 2

            # Получаем цвет и прозрачность
            red = int(poly[index] * 255)
            green = int(poly[index + 1] * 255)
            blue = int(poly[index + 2] * 255)
            alpha = int(poly[index + 3] * 255)

            # Рисуем многоугольник
            draw.polygon(vertices, (red, green, blue, alpha))

        del draw
        return image

    def getDifference(self, polygonData, method="MSE"):
        """
        Создаёт изображение и вычисляет отличие от эталона
        """

        image = self.polygonDataToImage(polygonData)

        if method == "MSE":
            return self.getMse(image)
        else:
            return 1.0 - self.getSsim(image)

    def plotImages(self, image, header=None):
        """
        Отображает эталонное изображение и результат рядом
        """

        fig = plt.figure("Сравнение изображений")
        if header:
            plt.suptitle(header)

        # Эталон
        fig.add_subplot(1, 2, 1)
        plt.imshow(self.refImage)
        self.ticksOff(plt)

        # Результат
        fig.add_subplot(1, 2, 2)
        plt.imshow(image)
        self.ticksOff(plt)

        return plt

    def saveImage(self, polygonData, imageFilePath, header=None):
        """
        Создаёт изображение из многоугольников
        и сохраняет сравнение с эталоном
        """

        image = self.polygonDataToImage(polygonData)
        self.plotImages(image, header)
        plt.savefig(imageFilePath)

    # Вспомогательные методы

    def toCv2(self, pil_image):
        """Преобразует Pillow-изображение в формат OpenCV"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def getMse(self, image):
        """Вычисляет среднеквадратичную ошибку (MSE)"""
        diff = self.toCv2(image).astype("float") - self.refImageCv2.astype("float")
        return np.sum(diff ** 2) / float(self.numPixels)

    def getSsim(self, image):
        """Вычисляет индекс структурного сходства (SSIM)"""
        return structural_similarity(
            self.toCv2(image),
            self.refImageCv2,
            multichannel=True
        )

    def list2Chunks(self, data, chunkSize):
        """Разбивает список на блоки фиксированного размера"""
        for i in range(0, len(data), chunkSize):
            yield data[i:i + chunkSize]

    def ticksOff(self, plot):
        """Отключает оси и подписи на графике"""
        plot.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False
        )
