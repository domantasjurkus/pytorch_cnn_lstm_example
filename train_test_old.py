import torch

#
# This file was used for training/testing without a dataloader and without batches
# Delete this file once training with dataloaders works
#

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

training_losses = []
testing_losses = []

def train(model, trainX, trainy, testX, testy, n_classes, epochs=10, masked=False):
    for epoch in range(epochs):
        total_loss = 0.0
        running_loss = 0.0

        for i, trainx in enumerate(trainX):
            inputs = trainx
            targets = trainy[i]
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            
            # batchify
            inputs.unsqueeze_(0)
            # targets.unsqueeze_(0)
            
            model.optimizer.zero_grad()
            outputs = model(inputs)

            loss = model.criterion(outputs, targets.long())
            loss.backward()
            model.optimizer.step()

            total_loss += loss.item()
            running_loss += loss.item()

            if i % 100 == 0:
                print('epoch: %d i: %d loss: %.3f' % (epoch, i, running_loss))
                running_loss = 0.0
        
        training_losses.append(total_loss)
        test(model, testX, testy, n_classes)

    print('Finished Training')
    print('Training losses:', training_losses)
    # print('Testing losses:', testing_losses)

def test(model, testX, testy, n_classes):
    correct = 0
    total = 0
    class_correct = [0]*n_classes
    class_total = [0]*n_classes

    running_loss = 0.0
    total_loss = 0.0
    confusion = torch.zeros([n_classes, n_classes], dtype=torch.int) # (class, guess)

    test_size = testy.size(0)

    with torch.no_grad():
        for i, testx in enumerate(testX):
            inputs = testx
            targets = testy[i]
            targets = targets.long()
            inputs, targets = inputs.to(device), targets.to(device)
            
            # batchify
            inputs.unsqueeze_(0)
            outputs = model(inputs)

            loss = model.criterion(outputs, targets)
            total_loss += loss.item()
            running_loss += loss.item()
            
            _, predicted_indexes = torch.max(outputs.data, 1)
            
            # bin predictions into confusion matrix
            # for j in range(len(inputs)):
            #     actual = targets[j].item()
            #     predicted = predicted_indexes[j].item()
            #     confusion[actual][predicted] += 1

            # sum up total correct
            batch_size = targets.size(0)
            total += batch_size
            correct_vector = (predicted_indexes == targets)
            correct += correct_vector.sum().item()
            
            # sum up per-class correct
            for j in range(len(targets)):
                target = targets[j]
                class_correct[target] += correct_vector[j].item()
                class_total[target] += 1

            if i % 5 == 0:
                print('test_size: %d/%d running loss: %.3f' % (i, test_size, running_loss))
                running_loss = 0.0
        
        testing_losses.append(total_loss)

    print('Correct predictions:', class_correct)
    print('Total test samples: ', class_total)
    print('Test accuracy: %d %%' % (100 * correct / total))
    # print(confusion)
