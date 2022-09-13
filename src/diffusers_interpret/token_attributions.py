import pandas as pd
import matplotlib.pyplot as plt


def plot(self, plot_type: str = 'barh', title: str = 'Token Attributions', **plot_kwargs) -> None:
    '''
    Plot the normalized token attributes to have a comparative view.
    Available plot types include bar chart, horizontal bar chart, and pie chart.
    '''
    tokens, attributions = list(
        zip(*self))  # TODO: this can be changed, depending on how we construct the class

    plot_kwargs = {'title': 'Token Attributions', **plot_kwargs}

    plt.title(title)

    if plot_type == 'bar':
        # Bar chart
        plt.bar(tokens, attributions)
        plt.xlabel('tokens')
        plt.ylabel('attribution value')

    elif plot_type == 'barh':
        # Horizontal bar chart
        plt.barh(tokens, attributions)
        plt.xlabel('attribution value')
        plt.ylabel('tokens')
        plt.gca().invert_yaxis()

    elif plot_type == 'pie':
        # Pie chart
        plt.pie(attributions,
                startangle=90,
                counterclock=False,
                #  explode = (attributions <= 3) * 0.5,
                labels=tokens,
                autopct='%1.1f%%',
                pctdistance=0.8)

    else:
        raise NotImplementedError(
            f"`plot_type = {plot_type} is not implemented. Choose one of: ['bar', 'barh', 'pie']")
