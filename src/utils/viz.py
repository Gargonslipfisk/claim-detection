import matplotlib.pyplot as plt

def hist_lbl_col(df, label, column):
    """
    Display a histogram for a column in a dataframe, splitting the data by label
    """
    plt.figure(figsize=(16,8))
    df[df[label] == 0][column].hist(label='0')
    df[df[label] == 1][column].hist(alpha=0.4, label='1')
    plt.title('Histograma de {} respecto a {}'.format(label, column))
    plt.legend()
    plt.show()