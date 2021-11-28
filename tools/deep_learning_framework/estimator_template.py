import tensorflow as tf

def create_model(params):
	# define model structure, loss and return value of network
	pass

def model_fn_builder(params):
	# return the model_fn used in Estimator
	def model_fn(features, labels, mode, params, config):
		'''
		features: first value returned from input_fn
		labels: second valus returned from input_fn
		mode: tf.estimator.ModeKeys
		params: dict
		config: Runconfig of Estimator
		'''
		create_model(params)
		if mode == tf.estimator.ModeKeys.PREDICT:
			#...
		elif mode == tf.estimator.ModeKeys.EVAL:
			#...
		elif mode == tf.estimator.ModeKeys.TRAIN:
			#...

		#...
		return tf.estimator.EstimatorSpec(...)

	return model_fn

def input_fn_builder(params):
	def input_fn():
		#...
		return features, labels
	return input_fn

def serving_input_receiver_fn():
	# 在此处 多说一些 关于 batch_features以及 receiver_tensor
    # 1. 首先 这两个 参数，相互之间 并没有 直接 的 关系（切记，没有直接关系，说明还是 有间接关系的）
    # 2. batch_features这个参数的格式必须 满足 model_fn中features参数格式
    # 2.1 关于值的格式，首先他必须是 tensor或者sparseTensor 或者 字典格式（value必须是tensor/sparsetensor）,然后features被传给model
    # 2.2 如果 features不是字典，则 该方法会自动将其封装为dict(视为一个样本)，并使用‘feature’作为key
    # 2.3 总结：model必须接受一个形如{'feature':tensor}的字典作为入参
    # 3.receiver_tensor 这个参数 是用来接收 请求 的 参数，该参数 一般可以 用一个 placeholder代替，后续经过各种变化，
    # 将receiver_tensor的值 转换为model_fn中features格式
    # 3.1 必须是 tensor或者sparseTensor 或者 字典格式（value必须是tensor/sparsetensor）

	return tf.estimator.export.ServingInputReceiver(batch_features, receiver_tensor)

estimator = tf.estimator.Estimator(
	model_fn=model_fn_builder,
	model_dir=PATH,
	params={"batch_size":BATCH_SIZE}
	)


# train(    input_fn,    hooks=None,    steps=None,    max_steps=None,    saving_listeners=None)
estimator.train(input_fn=input_fn_builder)

# evaluate(  input_fn,    steps=None,    hooks=None,    checkpoint_path=None,    name=None)
estimator.evaluate(input_fn=input_fn_builder)

# predict(    input_fn,    predict_keys=None,    hooks=None,    checkpoint_path=None,    yield_single_examples=True)
estimator.predict(...)


estimator.export_savedmodel('export_base',serving_input_receiver_fn=serving_input_receiver_fn)


# 1. tensorflow serving 
# 最好的 方式 是 使用 docker,详情请请参照：
# https://www.tensorflow.org/tfx/serving/docker
# https://www.tensorflow.org/tfx/serving/api_rest

# 2. 使用 tornado/flask
# steps：
# 1. load model
predictor = tf.contrib.predictor.from_saved_model(model_path)
# 2. predict
# input_params 格式必须 符合 serving_input_receiver_fn中入参
predict_result = predictor(input_params)

# 3. using tornado
class b_vxHandler(tornado.web.RequestHandler): 

    def post(self, version):
      
        try:
           #..... 接收参数并调用model
        except BaseException as err:
            self.finish(....)


application = tornado.web.Application([
    (r"/b/(?P<version>v\d+)", b_vxHandler),
])

if __name__ == "__main__":
    # tornado.options.parse_command_line()
    application.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

