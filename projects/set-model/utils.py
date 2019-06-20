

class Dataset():

	def __init__(self, dataset, batch_size_train=128, batch_size_test=128, 
				 stats_mean=None, stats_std=None):

		self.dataset_name = dataset

		# recover mean and std to normalize dataset
		if not stats_mean or not stats_std:
		    tempset = getattr(datasets, dataset)(root=data_dir, 
		    									 train=True,
		    									 transform=transforms.ToTensor(), 
		    									 download=True)
		    datastats_mean = tempset.train_data.float().mean().item()/255
		    datastats_std = tempset.train_data.float().std().item()/255
		    del tempset

		# set up transformations
	    transform = transforms.Compose(
	        [
	            transforms.ToTensor(),
	            transforms.Normalize((datastats_mean,), (datastats_std,)),
	        ]
	    )

	    # load train set
	    train_set = getattr(datasets, dataset)(root=data_dir, train=True,
	                                           transform=transform, 
	                                           download=True)
	    self.train_loader = DataLoader(dataset=train_set, 
	    							   batch_size=batch_size_train, 
	    							   shuffle=True)
	    

	    # load test set
	    test_set = getattr(datasets, dataset)(root=data_dir, train=False,
	                                          transform=transform, 
	                                          download=True)
	    self.test_loader = DataLoader(dataset=test_set, 
	    							  batch_size=batch_size_test, 
	    							  shuffle=False)