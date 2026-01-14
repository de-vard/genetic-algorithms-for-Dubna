# Image Reconstruction with Genetic Algorithm

Восстановление изображения с помощью **генетического алгоритма**
и набора полупрозрачных многоугольников (**DEAP**).

## Идея
Изображение аппроксимируется **треугольниками**.  
Генетический алгоритм минимизирует разницу (**MSE**) между эталоном и результатом.

## Стек
**Python** · DEAP · NumPy · Pillow · OpenCV · Matplotlib

## Запуск
```bash
python -m venv venv 
venv\Scripts\activate
pip install -r requirements.txt
python reconstruct-with-polygons.py
