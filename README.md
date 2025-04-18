# Лабораторная работа 2: 
## Описание
Исследование производительности параллельного умножения матриц с использованием многопоточности. Проект включает:
- Скрипт для тестирования разных размеров матриц и количества потоков
- Визуализацию результатов в виде heatmap
- Автоматическую валидацию результатов

## Требования
- Bash-совместимый терминал
- Python 3.8+
- Утилита `matrix_mult` (скомпилированная из исходников)
- Библиотеки: NumPy, Pandas, Matplotlib, Seaborn

## Запуск
```bash
./benchmark.sh
```

### Расширенные параметры 
  Параметр	Описание	По умолчанию
- --sizes	Размеры матриц (через пробел)	100 200 300 400 500
- --threads	Количество потоков	1 2 4 8
- --csv	Имя выходного CSV-файла	results.csv
- --plot	Имя файла графика	performance_plot.png
