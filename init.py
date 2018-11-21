from init.kmeans import kmeans_init

if __name__ == '__main__':
    def wizard():
        algos = [
            ('K-Means', kmeans_init)
        ]
        print('--------- Menu -----------')
        for i, names in enumerate(algos):
            print(f'\n{i}. {names[0]}\n')
        print('--------------------------')
        a = int(input('Choose which algorithm you want to execute:\n'))

        if a >= len(algos):
            print('\nInvalid choice. Try again:\n')
            wizard()
            return

        algos[a][1]()

    wizard()
