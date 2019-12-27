# converting from keras file to tflite file using tflite converte
import tensorflow.lite as lite



def convert(filename):
	converter = lite.TFLiteConverter.from_keras_model_file(filename)
	tflite_model = converter.convert()
	open('foodid_graph.tflite','wb').write(tflite_model) 

def main():
	convert('KERAS_FILE_TO_CONVERT_HERE')


if __name__ == '__main__':
	main()
