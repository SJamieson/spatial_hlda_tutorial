import matplotlib.pyplot as plt

scores = {'hint': [], 'no-hint': []}
with open('hierarchical_outputs/hlda-filters.scores', 'r') as in_file:
    for line in in_file:
        scores['no-hint'].append(float(line))
with open('hierarchical_outputs/hlda-filters-hints.scores', 'r') as in_file:
    for line in in_file:
        scores['hint'].append(float(line))

plt.plot(range(10,10000,10), scores['no-hint'], label='No Level Hints')
plt.plot(range(10,10000,10), scores['hint'], label='With Level Hints')
plt.legend()
plt.title('hLDA Convergence')
plt.ylabel('Gibbs Score')
plt.xlabel('Iterations')
plt.savefig('figures/hlda_convergence_hints.png')
plt.show()
