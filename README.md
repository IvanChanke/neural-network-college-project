# Нейронная сеть с нуля - Курсовая работа.

Это - репозиторий моей курсовой работы "Нейронная сеть для распознавания изображений". На данный момент я работаю над пользовательским интерфейсом, с помощью которого можно обучать создавать модель, обучать её логическим функциям и тестировать. В отдельном скрипте прописаны классы `Network`, `Layer` и `Node`, вместе составляющие реализацию математической модели полносвязной нейронной сети прямого распространения. Реализованная модель нейронной сети успешно протестирована на задачах обучения логическим функциям AND, OR, XOR.
Также имеется файл, хранящий в себе тренировачные данные для всех логических функций. Позднее туда будут добавлены данные для опроса модели.

Финальная цель проекта - разработать нейронную сеть, классифицирующую рукописные цифры из базы MNIST, сопровожденную графическим интерфейсом пользователя.

В репозитории присутствуют:
* Документы PDF - сопровождение ПО.
    * Техническое задание: ТЗ на курсовую.
    * Обзор аналогов и литературы.
    * Эскиз GUI. Описание основных функций.
    * Обоснование и описание темы курсовой.
    * Формулы: Краткий конспект реализованной математической модели обучения.
    
* Скрипты на Python:
    * network_code_eng - основной скрипт с классами для НС.
    * GUI_Scratch - черновик пользовательского интерфейса. 
    * Test_Script - скрипт, в котором я тестирую работу сети, экспериментирую с параметрами. Своеобразная песочница.
    * gui_processing - Файл, в котором я работаю над интерфейсом для создания/тестирования моделей, работающих с логическими функциями.
    
* Также в репозитории могут появляться и исчезать картинки, файлы обученных моделей и другие вспомогательные файлы.

(c) Иван Чанке 2020

НИУ ВШЭ\
ОП "Информатика и вычислительная техника"\
1 курс.
