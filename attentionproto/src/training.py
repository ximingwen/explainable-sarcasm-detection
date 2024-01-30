import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from model import AttentionProtoNet
import numpy as np
import pickle, argparse, json
# from sentence_transformers import SentenceTransformer, models
from sklearn_extra.cluster import KMedoids
from transformers import AutoTokenizer, AutoModel
from data_loader import custom_collate_fn, SarcasmDetection
from torch.utils.data import DataLoader
#####################  GPU Configs  #################################

# Function to calculate accuracy
def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100 * correct / total


# Selecting the GPU to work on
if __name__ == "__main__":
    
   
    np.random.seed(10)

    
    # Selecting the GPU to work on
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Initialize the argument parser
    parser = argparse.ArgumentParser()
    
    # Data loading params
    parser.add_argument("--dev_sample_percentage", type=float, default=0.1,
                        help="Percentage of the training data to use for validation")
    
    # Model Hyperparameters
    parser.add_argument("--embedding_dim", type=int, default=1024,
                        help="Dimensionality of character embedding (default: 128)")
    parser.add_argument("--filter_sizes", type=str, default="3,4,5",
                        help="Comma-separated filter sizes (default: '3,4,5')")
    parser.add_argument("--num_filters", type=int, default=128,
                        help="Number of filters per filter size (default: 128)")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.5,
                        help="Dropout keep probability (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", type=float, default=0.5,
                        help="L2 regularization lambda (default: 0.0)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=60,
                        help="Batch Size (default: 64)")
    parser.add_argument("--num_epochs", type=int, default=4000,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--evaluate_every", type=int, default=100,
                        help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="Save model after this many steps (default: 100)")
    parser.add_argument("--num_checkpoints", type=int, default=5,
                        help="Number of checkpoints to store (default: 5)")
    
    # Misc Parameters
    parser.add_argument("--allow_soft_placement", action="store_true",
                        help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", action="store_true",
                        help="Log placement of ops on devices")

    parser.add_argument("--l1", type=float, default = 0.1)
    parser.add_argument("--l2", type=float, default = 0.1)
    parser.add_argument("--k_protos", type=int, default = 16)
    parser.add_argument("--accumulation_steps", type=int, default = 30)
    parser.add_argument("--scale", type=float, default = 5)
    parser.add_argument("--threshold", type=float, default = 0.8)
    
    parser.add_argument("--num_of_sentence", type=int, default = 4)
    parser.add_argument("--n_layers", type=int, default = 1)
    parser.add_argument("--n_heads", type=int, default = 1)
    parser.add_argument("--pf_dim", type=int, default = 2048)
    parser.add_argument("--encoder_dropout", type=int, default = 0.1)
    parser.add_argument("--device", type=str, default ="cuda:0")
    parser.add_argument("--language_model", type=str, default ="roberta-base")


    
    # Parse the command-line arguments
    args = parser.parse_args()


    
    num_of_sentence = args.num_of_sentence
    n_layers = args.n_layers
    n_heads = args.n_heads
    pf_dim = args.pf_dim
    encoder_dropout = args.encoder_dropout
    device = args.device
    language_model = args.language_model
    batch_size = args.batch_size


    out_dir = "./output/"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(os.path.join(out_dir, 'arguments.txt'), 'w+') as out_f:
        out_f.write(str(args))

    
    print("loading data...")
    revs = pickle.load(open("./mainbalancedpickle-new.p","rb"))
    
    print("data loaded!")# Load data
    
   
    
    x_text = []
    y = []
    
    x_test = []
    y_test = []
    
    for i in range(len(revs)):
        if revs[i]['split']==1:
            x_text.append(revs[i]['combined'])
            y.append(revs[i]['label'])
        else:
            x_test.append(revs[i]['combined'])
            y_test.append(revs[i]['label'])  
    
 
    
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = np.asarray(x_text)[shuffle_indices]
    y_shuffled = np.asarray(y)[shuffle_indices]

    
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    
    dev_sample_index = -1 * int(args.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    
    
    x_train = np.asarray(x_train)
    x_dev = np.asarray(x_dev)
    x_test = np.array(x_test)
    y_train = np.asarray(y_train)
    y_dev = np.asarray(y_dev)
    y_test = np.array(y_test)
    
    
  
    # Training
   

    print(language_model)

    # Create Dataset
    train_dataset = SarcasmDetection(x_train, y_train, language_model)
    dev_dataset = SarcasmDetection(x_dev, y_dev, language_model)
    test_dataset = SarcasmDetection(x_test, y_test, language_model)


    # Create DataLoader

   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)  
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)


    
    vect_size = 384

    classifier = AttentionProtoNet(
        sequence_length=100,
        num_classes=len(y_train[0]) ,
        embedding_model = language_model,
        l2_reg_lambda=args.l2_reg_lambda,
        dropout_keep_prob = args.dropout_keep_prob,
        k_protos = args.k_protos,
        embedding_dim = 768,
        num_of_sentence = num_of_sentence,
        n_layers = n_layers, 
        n_heads = n_heads, 
        pf_dim = pf_dim, 
        encoder_dropout = encoder_dropout,
        device = device).to(device)
    
    
    sample_sentences_vects = []
    datapoint = 0
    for i, inputs in tqdm(enumerate(train_loader)):
        x_batch, y_batch = inputs
        vect = classifier.embed(x_batch).detach().cpu().numpy()
        sample_sentences_vects.append(vect)
        datapoint+=batch_size

        if datapoint>150:
            break

    

    # # random.shuffle(x_text)
    # sample_sentences = x_text[:50]
    # sample_sentences_vects = []
    # for i in range(1):
    #     batch = sample_sentences[i * 50:(i + 1) * 50]
    #     vect = classifier.embed(batch)
    #     sample_sentences_vects.append(vect)

  
    sample_sentences_vect = np.vstack(sample_sentences_vects)
    print(sample_sentences_vect.shape)
    kmedoids = KMedoids(n_clusters=args.k_protos, random_state=0).fit(sample_sentences_vect)
    sent_cents = kmedoids.cluster_centers_
    
    
    classifier.init_prototypelayer(sent_cents)
    
    # Define optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    
    # Define criterion (loss function)
    criterion = criterion = nn.CrossEntropyLoss()
    
    # Define metrics for training, validation, and test accuracy
    # In PyTorch, you typically calculate accuracy manually during the training loop,
    # as there is no direct equivalent to tf.keras.metrics.CategoricalAccuracy
    
    def accuracy(output, target):
        """Compute the accuracy, given the output and target tensors."""
        predictions = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = predictions.eq(target.view_as(predictions)).sum().item()
        return correct / output.size(0)
    
    # Directory creation
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
   
    # Training loop. For each batch...

    accumulation_steps = 30
    

    train_loss = []
    train_acc = []
    dev_loss  = []
    dev_acc = []
    test_acc = []
    
    train_res_div_loss = []
    train_user_div_loss = []
    train_acc_loss = []

    dev_res_div_loss = []
    dev_user_div_loss = []
    dev_acc_loss = []
    best_loss_so_far = float("inf")

    
    for epoch in range(args.num_epochs):
        
        epoch_loss = 0
        total_correct = 0
        total_samples = 0
    
        classifier.train()  # Set model to training mode
    
        for i, inputs in tqdm(enumerate(train_loader)):
            x_batch, y_batch = inputs
            y_batch = torch.tensor(y_batch)
    
         
    
            # Forward pass
            predictions = classifier(x_batch)
    
            loss = criterion(predictions, y_batch)
            loss = loss/accumulation_steps
            epoch_loss += loss.item()* x_batch.size(0)
    
            # Backward pass and optimize
            loss.backward()
    
            # In PyTorch, gradients are accumulated by default, so you control this with optimizer steps
            if (i + 1) % accumulation_steps == 0 or i == len(train_loader) // args.batch_size - 1:
                
                optimizer.step()
                optimizer.zero_grad()
                print(f"Epoch: {epoch + 1}, Batch: {i + 1}/{len(train_loader)}, Loss: {epoch_loss / (args.batch_size * accumulation_steps)}, Accuracy: {train_correct / train_total}")
                accumulated_loss = 0


            # Update accuracy
            correct, total = calculate_accuracy(outputs, targets)
            total_correct += correct
            total_samples += total

        epoch_accuracy =  total_correct / total_samples
        epoch_loss /= len(train_loader)
        print(f"Epoch: {epoch+1}, epoch Loss: {epoch_loss }  train accuracy:{epoch_accuracy}\n")
        
        train_loss.append(epoch_loss )
        train_acc.append(epoch_accuracy)
            

        valid_loss = 0
        total_correct = 0
        total_samples = 0
        
        ProtoCNN.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():  # Disable gradient computation
            for x_batch, y_batch in tqdm(dev_loader):
                # Convert batches to PyTorch tensors
                y_batch = torch.tensor(y_batch)
        
                # Forward pass
                predictions = classifier(x_batch)
        
                # Compute accuracy loss
                loss = criterion(predictions, y_batch)
                
                valid_loss += loss.item() * x_batch.size(0)     
        
                # Update validation accuracy
                correct, total = calculate_accuracy(predictions, y_batch)
                total_correct += correct
                total_samples += total
            
            # Calculate average losses and accuracy
            valid_loss /= len(dev_loader)
            valid_accuracy = total_correct / total_samples
          
    
            dev_loss.append(valid_loss)
            dev_acc.append(valid_accuracy)   
            print(
                f"Epoch: {epoch + 1}, Valid Loss: {valid_loss}  valid accuracy:{valid_accuracy}\n")

    

            if valid_loss < best_loss_so_far or epoch%10==0:
            
                print("find better loss")
                ProtoCNN.save_weights(os.path.join(out_dir, str(epoch)+"_best_classifier.ckpt"))
                pickle.dump(opt.get_weights(), open(os.path.join(out_dir, str(epoch)+"_optimizer.pt"), "wb+"))
                best_loss_so_far = valid_loss
            
            
          
                total_correct = 0
                total_samples = 0
                with torch.no_grad(): 
                    for x_batch, author_batch, topic_batch, y_batch in tqdm(test_loader):
        
                       
                        author_batch = np.asarray(author_batch)
                        topic_batch = np.asarray(topic_batch)
                        y_batch = np.asarray(y_batch)
            
                        #predictions, _, _ = ProtoCNN([x_batch, author_batch, topic_batch], training=False)
                        predictions = classifier(x_batch)
                        correct, total = calculate_accuracy(predictions, y_batch)
                        total_correct += correct
                        total_samples += total
            
            
                test_accuracy = total_correct / total_samples
                test_acc.append((epoch+1, test_accuracy))
    
                print(f"Epoch: {epoch + 1},   test accuracy:{test_accuracy_metric.result().numpy()}")
    

        with open(out_dir+'train_losses.pkl', 'wb') as file:
            pickle.dump((train_loss, train_acc_loss,train_user_div_loss, train_res_div_loss, train_acc), file)

        with open(out_dir+'valid_losses.pkl', 'wb') as file:
            pickle.dump((dev_loss, dev_acc_loss, dev_user_div_loss, dev_res_div_loss, dev_acc), file)

        with open(out_dir+'test_acc.pkl', 'wb') as file:
            pickle.dump(test_acc, file)

            
            
    
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))  # Optional: set figure size
        plt.plot(train_loss, label='Training Loss')
        plt.plot(dev_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Save the plot as an image
        plt.savefig('training_validation_loss.png')
        
        # Optionally, display the plot
        plt.show()
