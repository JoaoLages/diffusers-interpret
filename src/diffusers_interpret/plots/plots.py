from typing import Literal, get_args, get_origin
import inspect
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

def enforce_literals(function):
    '''
    Dynamically enforces the use of literals for function arguments.
    Use the `inspect` module to get the previous frame to extract the provided arguments.
    Use the function's attribute `__annotations__` to get the expected types.
    '''
    frame = inspect.stack()[1].frame
    *_, parameters = inspect.getargvalues(frame)
    for name, literal in function.__annotations__.items():
        if get_origin(literal) is Literal and name in parameters:
            value = parameters[name]
            assert value in get_args(literal), f"'{value}' is invalid - valid options are {get_args(literal)}"

# Plot types can only be one of these categories
_TYPES = Literal['bar', 'barh', 'pie']

def plot(self, type: _TYPES = 'bar', title: str = 'Token Attributions', rot: int = 60):
    '''
    Plot the normalized token attributes to have a comparative view.
    Available plot types include bar chart, horizontal bar chart, and pie chart.
    '''
    # Dynamic assertation
    enforce_literals(plot)

    # Convert list of tuples to a dataframe
    df = pd.DataFrame(self, columns=['Tokens', 'percent']).set_index('Tokens')

    # Bar chart
    if type == 'bar':
        df.plot.bar(ylabel = 'percent',
            title = title,
            legend = False,
            rot = rot);
    
    # Horizontal bar chart
    elif type == 'barh':
        df.plot.barh(ylabel = 'percent',
            title = title,
            legend = False,
            rot = 0);

    # Pie chart
    elif type == 'pie':
        df.plot.pie(y = 'percent',
            title = title,
            legend = False,
            autopct = '%1.1f%%',
            figsize = (8, 8));

        

