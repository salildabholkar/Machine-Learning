from init.kmeans import kmeans_init
from init.naive_bayes import naive_bayes_init

def wizard():
    algos = [
        ('K-Means', kmeans_init),
        ('Naive Bayes', naive_bayes_init)
    ]
    print('--------- Menu -----------\n')
    for i, names in enumerate(algos):
        print(f'{i}. {names[0]}')
    print('\n--------------------------')
    a = int(input('Choose which algorithm you want to execute:\n'))

    if a >= len(algos):
        print('\nInvalid choice. Try again:\n')
        wizard()
        return

    algos[a][1]()

if __name__ == '__main__':
    wizard()
