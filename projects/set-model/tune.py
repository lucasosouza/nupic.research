

class Tune(tune.Trainable):
	"""
	ray.tune trainable class for SET. 
	Adaptable to any pytorch module
	"""

	def __init__(self, model, dataset, config=None, logger_create=None):
		tune.Trainable.__init__(self, config=config, logger_creator=logger_creator)
		self.model = model
		self.dataset = dataset

	def _setup(self, config):
		self.model.setup(config)

	def _train(self):
		log = self.model.run_trial(self.dataset)
		return log

	def _save(self, checkpoint_dir):
		self.model.save(checkpoint_dir)

	def _restore(self, checkpoint):
		self.model.restore(checkpoint)










