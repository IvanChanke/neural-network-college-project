# Нейронная сеть с нуля - Курсовая работа.
### Актуально на 11.04. 

Это - репозиторий моей курсовой работы "Нейронная сеть для распознавания изображений". На данный момент в отдельном скрипте прописаны классы `Network`, `Layer` и `Node`, вместе составляющие реализацию математической модели полносвязной нейронной сети прямого распространения. Реализованная модель нейронной сети успешно протестирована на задачах обучения логическим функциям AND, OR, XOR.
Для обучения и тестирования сети на примере этих логических функций разработано приложение LogicNetApp. Приложение работает.
На данный момент я тестирую работу сети непосредственно на задаче распознавания изображений из MNIST. Для этих целей написано приложение DigitRecApp.
Также имеется файл, хранящий в себе тренировачные данные для всех логических функций. В репозиторий также загружена база MNIST в формате png.

Финальная цель проекта - разработать нейронную сеть, классифицирующую рукописные цифры из базы MNIST, сопровожденную графическим интерфейсом пользователя.

В репозитории присутствуют:
* Документы PDF - сопровождение ПО.
    * Техническое задание: ТЗ на курсовую.
    * Обзор аналогов и литературы.
    * Эскиз GUI. Описание основных функций.
    * Обоснование и описание темы курсовой.
    * Формулы: Краткий конспект реализованной математической модели обучения.
    * Прототип ПО. Черновое руководство пользователя.
    
* Скрипты на Python:
    * ptron - основной скрипт-библиотека с классами для НС.
    * GUI_Scratch - черновик пользовательского интерфейса. 
    * Test_Script - скрипт, в котором я тестирую работу сети, экспериментирую с параметрами. Своеобразная песочница.
    * LogicNetApp - Файл для создания/тестирования моделей, работающих с логическими функциями.
    * dtrain - скрипт-библиотека с тренировачными данными для логических функций.
    * annotations - скрипт с текстом, используемым приложением LogicNetApp.
    * annotations2 - скрипт с текстом, используемым приложением DigitRecApp.
    * DigitNetApp - скрипт для тестирования распознавания изображений. Неокончательная версия.
    * digit_recognition - черновик.

* XOR6000 - обученная на 6000 итерациях модель, реализующая логический гейт "исключающее или".

(c) Иван Чанке 2020

НИУ ВШЭ\
ОП "Информатика и вычислительная техника"\
1 курс.
