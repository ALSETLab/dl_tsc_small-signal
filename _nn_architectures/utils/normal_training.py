from sklearn.model_selection import train_test_split
import os

def normal_training(classifier, x_train, y_train, x_test, y_test, x_valid, y_valid, nb_classes):

    print('\n{t:-^30}'.format(t = ''))
    print("Normal training")
    print('{t:-^30}'.format(t = ''))

    #x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size = 0.333, random_state = 42)

    #print(f'x_train: {x_train.shape}')
    #print(f'x_test: {x_test.shape}')
    #print(f'y_train: {y_train.shape}')
    #print(f'y_test: {y_test.shape}')

    # Training classifier`
    classifier.fit(x_train, y_train, x_valid, y_valid)

    df_metrics = classifier.predict(x_test, y_test, x_train, y_train, nb_classes)

    df_metrics.to_csv(os.path.join(classifier.output_directory, 'results_normal.csv'), index = False)

    print(df_metrics)
