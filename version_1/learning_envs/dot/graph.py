import matplotlib.pyplot as plt
import os, json



fitnesses = []
cwd = os.getcwd()
files = os.listdir(cwd + "/generations")

for file in files:
    if file.startswith("generation_"):
        with open('generations/'+file, 'r') as f:
            data = json.load(f)
        
        top = max([ai[1] for ai in data])
        fitnesses.append(top)


plt.plot(fitnesses)
plt.ylabel('Fitness')
plt.xlabel('Generation')
plt.show()