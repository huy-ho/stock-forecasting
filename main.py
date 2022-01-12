from new_model import PreProcessed_Model
def main():
    new_model = PreProcessed_Model('NVDA')
    new_model.build_model()
    new_model.preprocessed_data()
    print(new_model.predict('AAPL'))
    print(new_model.predict('AMD'))
    print(new_model.predict('PANW'))
    print(new_model.predict('T'))
    print(new_model.predict('GME'))
    print(new_model.predict('CSCO'))
    
if __name__ =='__main__':
        main()
    
