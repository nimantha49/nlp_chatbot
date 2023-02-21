import random
import json
import torch
from Neural import NeuralNet
from nltk_code import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('commands.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Cyd"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.80:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I'm sorry, I'm not sure what you mean. Can you ask the question in a different way please? If not you will certainly find it in our website."


if __name__ == "__main__":
    print(f"{bot_name}: Hi! I'm Cyd, SLTC Info Center ChatBot. How Can I Help you? \n     (TYPE 'quit' TO EXIT)\n")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            print(f'{bot_name}: Have a nice day, Good Bye!')
            break

        resp = get_response(sentence)
        print(f"{bot_name}: {resp}")
    