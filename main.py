import model
import data_generator
import os
import pylib as py
import json

#Add code for last entry data file. (For systematic number of LOGS and CHECKPOINTS.
#Add code for parsing through JSON files to find already trained code.

#py.arg('--prev_train_data', default = "./Previous Train Data")
py.arg('--dataset', default = "./Data")
py.arg('--batch_size', type = int, default =4)
py.arg('--epochs', type = int, default = 100)
py.arg('--lr', type = float, default = 0.002)
py.arg('--image_size', type = int, default = 256)
py.arg('--n_channels', type = int, default = 3)
py.arg('--shuffle_data', type = bool, default = True)
py.arg('--bottleneck_size', type = int, default = None)
py.arg('--loss_weight', type = float, default = 0.8)
py.arg('--checkpoint_dir', default = None)
py.arg('--tensorboard_dir', default = "./logs")
py.arg('--prev_checkpoint', default = None)
py.arg('--prev_tensorboard', default = None)
args = py.args()

##Finding previously trained model. [CODE NOT COMPLETED]
#json_directory = "args.prev_train_data"
#json_list = os.listdir(json_directory)
#for json_file in json_list:
#  with open(os.path.join(json_directory, json_file)) as f:
#    metadata = json.load(f)
#    if (metadata['successful_completion'] and metadata['dataset'] == args.dataset and metadata['batch_size'] == args.batch_size and metadata['lr'] == args.lr and metadata['image_size'] == args.image_size and metadata['n_channels'] =#= args.n_channels and metadata['bottleneck_size'] == args.bottleneck_size and metadata['loss_weight'] == args.loss_weight):
#      if (metadata['epochs'] < args.epochs):
#        args.epochs -= metadata['epochs']
        

AnoVAEGAN1 = model.AnoVAEGAN((args.image_size, args.image_size, args.n_channels))
input_shape, custom_bottleneck_size, generator_optimizer, discriminator_optimizer, batch_size, epochs, loss_weight, checkpoint_dir, learning_rate, log_path, checkpoint_prefix = AnoVAEGAN1.change_params(custom_bottleneck_size = args.bottleneck_size, 
                         batch_size = args.batch_size, 
                         epochs = args.epochs, 
                         loss_weight = args.loss_weight, 
                         checkpoint_dir = args.checkpoint_dir,
                         logs = args.tensorboard_dir,
                         )
                         
if (args.prev_checkpoint != None):
    AnoVAEGAN1.load_model_checkpoint(args.prev_checkpoint)

train_path_list = os.listdir(args.dataset + '/train')
test_path_list = os.listdir(args.dataset + '/test')

train_generator = data_generator.DataGenerator(list_IDs=train_path_list, 
                                               directory=args.dataset + '/train', 
                                               batch_size=args.batch_size, 
                                               image_size=(args.image_size, args.image_size), 
                                               n_channels = args.n_channels, 
                                               shuffle = args.shuffle_data)

test_generator = data_generator.DataGenerator(list_IDs=test_path_list, 
                                              directory=args.dataset + '/test', 
                                              batch_size=len(test_path_list))

# generated_images = AnoVAEGAN1.generator(train_generator.__getitem__(0), training = True)
# # print(generated_images.shape)

#Adding data to JSON file.
#metadata = {}
#metadata['dataset'] = args.dataset
#metadata['batch_size'] = batch_size
#metadata['epochs'] = epochs
#metadata['lr'] = learning_rate
#metadata['image_size'] = args.image_size
#metadata['n_channels'] = args.n_channels
#metadata['shuffle_data'] = args.shuffle_data
#metadata['bottleneck_size'] = args.bottleneck_size
#metadata['loss_weight'] = loss_weight
#metadata['checkpoint_dir'] = checkpoint_dr
#metadata['tensorboards_dir'] = tensorboard_dir
#metadata['tensorboard_log_name'] = log_path
#metadata['checkpoint_name'] = checkpoint_prefix
#metadata['successful_completion'] = False

#save_name = os.path.join("./Previous Train Data", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#with open(save_name, "w") as f:
#  json.dump(metadata, f)

# AnoVAEGAN1.printModelSummary()
AnoVAEGAN1.train(train_generator, args.epochs, test_generator)

#with open(save_name, "r") as f:
#  metadata = json.load(f)
#  metadata['successful_completion'] = True

#with open(save_name, "w") as f:
#  json.dump(metadata, f)