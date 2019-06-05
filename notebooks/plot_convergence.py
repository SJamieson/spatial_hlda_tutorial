import matplotlib.pyplot as plt

scores = {'hint': [], 'no-hint': []}
with open('hierarchical_outputs/hlda-filters.scores', 'r') as in_file:
    for line in in_file:
        scores['no-hint'].append(float(line))
with open('hierarchical_outputs/hlda-filters-hints.scores', 'r') as in_file:
    for line in in_file:
        scores['hint'].append(float(line))

plt.figure(1)
plt.plot(range(10,10000,10), scores['no-hint'], label='No Level Hints')
plt.plot(range(10,10000,10), scores['hint'], label='With Level Hints')
plt.legend()
plt.title('hLDA Convergence')
plt.ylabel('Gibbs Score')
plt.xlabel('Iterations')
plt.semilogx()
plt.savefig('figures/hlda_convergence_hints.png')
plt.show()


scores = {'hint': [], 'no-hint': []}
with open('hierarchical_outputs/hlda-sivic.scores', 'r') as in_file:
    for line in in_file:
        scores['no-hint'].append(float(line))
with open('hierarchical_outputs/hlda-sivic-hints.scores', 'r') as in_file:
    for line in in_file:
        scores['hint'].append(float(line))

plt.figure(2)
plt.plot(range(10,10000,10), scores['no-hint'], label='No Level Hints')
plt.plot(range(10,10000,10), scores['hint'], label='With Level Hints')
plt.legend()
plt.title('hLDA Convergence')
plt.ylabel('Gibbs Score')
plt.xlabel('Iterations')
plt.semilogx()
plt.savefig('figures/hlda_sivic_convergence_hints.png')
plt.show()


scores = {'hint': [], 'no-hint': []}
with open('hierarchical_outputs/hlda-sivic.scores', 'r') as in_file:
    for line in in_file:
        scores['no-hint'].append(float(line))
with open('hierarchical_outputs/hlda-sivic-hints-v2.scores', 'r') as in_file:
    for line in in_file:
        scores['hint'].append(float(line))

plt.figure(3)
plt.plot(range(10,10000,10), scores['no-hint'], label='No Level Hints')
plt.plot(range(10,10000,10), scores['hint'], label='With Level Hints')
plt.legend()
plt.title('hLDA Convergence')
plt.ylabel('Gibbs Score')
plt.xlabel('Iterations')
plt.semilogx()
plt.savefig('figures/hlda_sivicv2_convergence_hints.png')
plt.show()


scores = {'hint': [], 'no-hint': []}
with open('hierarchical_outputs/hlda-filters.scores', 'r') as in_file:
    for line in in_file:
        scores['no-hint'].append(float(line))
with open('hierarchical_outputs/hlda-filters-hints-v2.scores', 'r') as in_file:
    for line in in_file:
        scores['hint'].append(float(line))

plt.figure(4)
plt.plot(range(10,10000,10), scores['no-hint'], label='No Level Hints')
plt.plot(range(10,len(scores['hint'])*10+10,10), scores['hint'], label='With Level Hints')
plt.legend()
plt.title('hLDA Convergence')
plt.ylabel('Gibbs Score')
plt.xlabel('Iterations')
plt.semilogx()
plt.savefig('figures/hlda_filtersv2_convergence_hints.png')
plt.show()