from typing import Tuple, List, Any, Union
import matplotlib.pyplot as plt


class TokenAttributions(list):
    def __init__(self, token_attributions: List[Tuple[str, float]]) -> None:
        super().__init__(token_attributions)
        self.token_attributions = token_attributions

        # Calculate normalized
        total = sum([attr for _, attr in token_attributions])
        self.normalized = [
            (token, round(100 * attr / total, 3))
            for token, attr in token_attributions
        ]

    def __getitem__(self, item: Union[str, int]) -> Any:
        return getattr(self, item) if isinstance(item, str) else self.token_attributions[item]

    def __setitem__(self, key: Union[str, int], value: Any) -> None:
        setattr(self, key, value)

    def plot(self, plot_type: str = 'barh', normalize: bool = False, **plot_kwargs) -> None:
        '''
        Plot the token attributions to have a comparative view.
        Available plot types include bar chart, horizontal bar chart, and pie chart.
        '''

        attrs = self.normalized if normalize else self.token_attributions
        tokens, attributions = list(zip(*attrs))
        prefix = 'normalized ' if normalize else ''

        # get arguments from plot_kwargs
        xlabel = plot_kwargs.get('xlabel')
        ylabel = plot_kwargs.get('ylabel')
        title = plot_kwargs.get('title') or f'{prefix.title()}Token Attributions'

        if plot_type == 'bar':
            # Bar chart
            plt.bar(tokens, attributions)
            plt.xlabel(xlabel or 'tokens')
            plt.ylabel(ylabel or f'{prefix}attribution value')

        elif plot_type == 'barh':
            # Horizontal bar chart
            plt.barh(tokens, attributions)
            plt.xlabel(xlabel or f'{prefix}attribution value')
            plt.ylabel(ylabel or 'tokens')
            plt.gca().invert_yaxis() # to have the order of tokens from top to bottom

        elif plot_type == 'pie':
            # Pie chart
            plot_kwargs = {
                'startangle': 90, 'counterclock': False, 'labels': tokens, 
                'autopct': '%1.1f%%', 'pctdistance': 0.8,
                **plot_kwargs    
            }
            plt.pie(attributions, **plot_kwargs)
            if xlabel:
              plt.xlabel(xlabel)
            if ylabel:
              plt.ylabel(ylabel)

        else:
            raise NotImplementedError(
                f"`plot_type={plot_type}` is not implemented. Choose one of: ['bar', 'barh', 'pie']"
            )
        
        # set title and show
        plt.title(title)
        plt.show()
