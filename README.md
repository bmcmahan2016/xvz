# X-VisualiZation (XVZ) Callbacks for TensorFlow 
TensorFlow callbacks for visualizing weights and activations in deep convolutional neural network models

I built this API for a few TensorFlow callbacks to make it easier to debug deep CNN models during training. XVZ contains three custom callbacks that allow for the weights and activations to be saved to a tensorboard at the end of each training epoch.

Use the VisuzlizeConvWeights() callback to see the weights of all convolutional filters at each training epoch.
Use the VisualizeDenseWeights() callback to see the weights of all fully connected layers at each training epcoh.
Use VisualizeActivations() callback to see the activities of each convolutional filter in response to a randomly chosen image. 
