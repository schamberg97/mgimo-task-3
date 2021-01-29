import pandas as pd #Import pandas library
import PySimpleGUIQt as sg
from main import read_data, implement_machine_learning

class Loader:
    def __init__(self):
        self.df = None

    def load_dataset(self):
        self.df = read_data()


def main():    

    print("Оценка целесообразности звонка потенциальному клиенту")

    df_holder = Loader()
    df_holder.load_dataset()
    lda = implement_machine_learning(df_holder.df)

    values = {
        "age": int(input("Введите возраст потенциального клиента: ")),
        "balance": int(input("Введите баланс потенциального клиента: ")),
        "housing": bool(int(input("Введите 1, если у клиента есть жильё или 0, если его нет: "))),
        "job_student": int(input("Введите 1, если клиент - студент (или 0, если нет): ")),
        "job_unemployed": int(input("Введите 1, если клиент - безработный (или 0, если нет): ")),
        "marital_divorced": int(input("Введите 1, если клиент - в разводе (или 0, если нет): ")),
        #"marital_married": int(input("Введите 1, если клиент - в браке (или 0, если нет): ")),
        "marital_single": int(input("Введите 1, если клиент - не состоит в браке (или 0, если состоит): ")),
        "education_secondary": int(input("Введите 1, если клиент закончил среднее образование (или 0, если не закончил): ")),
        "education_tertiary": int(input("Введите 1, если клиент имеет высшее образование (или 0, если не имеет): ")),
        "default": bool(int(input("Введите 1, если у клиента был дефолт по кредиту или 0, если его не было: "))),
        "loan": bool(int(input("Введите 1, если у клиента есть долги или 0, если их нету: "))),
        "campaign": int(input("Campaign: "))
    }
    if (values["marital_divorced"] == 1):
        values["marital_married"] = 0
    elif (values["marital_divorced"] == 0):
        if (values["marital_single"] == 1):
            values["marital_married"] = 0
        else:
            values["marital_married"] = 1
        
    print(values)
    res = lda.predict(pd.DataFrame(values, index=[0]))
    whatToPrint = "Вам не следует делать предложение данному клиенту"
    text_color = "red"
    if (res > 0.5):
        whatToPrint = "Свяжитесь и предложите данному клиенту банковские продукты"
        text_color = "green"
    sg.popup_ok(whatToPrint, title = "Результат", font = "Helvetica 12", background_color="white", text_color=text_color, keep_on_top=True)

    main()

main()