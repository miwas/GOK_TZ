"""
from bokeh.models import ColumnDataSource #, Title, LabelSet, FactorRange
from bokeh.models import Select, MultiSelect, MultiChoice
from bokeh.layouts import column, row
from bokeh.io import curdoc

from utils import load_df, filter_df, make_capital_df

from plotting import create_points
from columns import ENG_TO_RU
"""

import numpy as np
import pandas as pd

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure, show
# from bokeh.io import output_notebook
from bokeh.io import curdoc

from pyod.models.iforest import IForest

# Load & filter data
"""

df = load_df('data/Report_Medtronic_Total_CRANIAL_and_SPINE.csv.gz')

df_cap = df[df['type_group_2']=='Капитальное оборудование']

df_cap_final = make_capital_df(df_cap, freq='Q')
#df_cap_final.head()

"""
SENSOR = "1309"
NAV = 0 # 1200
sensors = pd.read_parquet("./data/interim/sensors.parq")

out_sensor = sensors.loc[:,SENSOR] # TODO: to widget
print(out_sensor.shape)

out_sensor.replace(NAV, np.nan, inplace=True)
out_sensor.dropna(inplace=True)
#print(out_sensor.isnull().sum())
#out_sensor.describe()

# Initialize an instance with default parameters
iforest = IForest()

# Generate outlier labels
labels = iforest.fit(out_sensor.values.reshape(-1, 1))

probs = iforest.predict_proba(out_sensor.values.reshape(-1, 1))
print(probs)

# plt.hist(probs[:,1])

is_outlier = probs[:,1] > 0.8 # TODO: to widget
outliers = out_sensor[is_outlier]
#print(len(outliers))
# sum(is_outlier)

#[START CREATE PLOT]
"""

source_cap = ColumnDataSource(df_cap_final)
print("SOURCE COLUMNS: ", source_cap.column_names)

CORPS = df_cap.corp.unique()
print("# of CORPS: ", len(CORPS))

X_RANGE = list(df_cap_final.date_period.unique())

title_plot, p = create_points(source_cap, CORPS, X_RANGE)

"""

source = ColumnDataSource(out_sensor.reset_index())
source_out = ColumnDataSource(outliers.reset_index())

# print(out_sensor.index.min())
# print(out_sensor.index.max())


p = figure(height=600, width=2000,
           # tools="xpan", toolbar_location=None,
           x_axis_type="datetime", x_axis_location="above",
           # background_fill_color="#efefef",
           # x_range=(dates[1500], dates[2500])
           x_range=(out_sensor.index.min(), out_sensor.index.max())
          )

#p.line('date', 'close', source=source)
p.line('TIME', SENSOR, source=source) # # TODO: to widget
p.circle('TIME', SENSOR, source=source_out, color="red") # # TODO: to widget
p.yaxis.axis_label = 'Value'

select = figure(title="Drag the middle and edges of the selection box to change the range above",
                height=130, width=2000, # y_range=p.y_range,
                x_axis_type="datetime", y_axis_type=None,
                tools="",
                toolbar_location=None,
                # background_fill_color="#efefef"
                )

range_tool = RangeTool(x_range=p.x_range)
range_tool.overlay.fill_color = "navy"
range_tool.overlay.fill_alpha = 0.2

select.line('TIME', SENSOR, source=source) # TODO: to widget
select.ygrid.grid_line_color = None
select.add_tools(range_tool)
select.toolbar.active_multi = range_tool



# rng = np.random.default_rng()
# x = rng.normal(loc=0, scale=1, size=1000)

histplot = figure(width=670, height=300, toolbar_location=None,
           title="Distribution of values")

# Histogram
# bins = np.linspace(-3, 3, 40)
# Find the square root of the length
n_bins = np.sqrt(len(out_sensor))

# Cast to an integer
n_bins = int(n_bins)

hist, edges = np.histogram(out_sensor.dropna().values, density=True, bins=n_bins)
histplot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
         fill_color="skyblue", line_color="white",
         # legend_label="1000 random samples"
         )

# Probability density function
# x = np.linspace(-3.0, 3.0, 100)
# pdf = np.exp(-0.5*x**2) / np.sqrt(2.0*np.pi)
# histplot.line(x, pdf, line_width=2, line_color="navy",
#        legend_label="Probability Density Function")

histplot.y_range.start = 0
histplot.xaxis.axis_label = "x"
histplot.yaxis.axis_label = "PDF(x)"

# show(p)

# show(column(p, select), notebook_handle=True)

#[END CREATE PLOT]

#[START CREATE WIDGETS]
"""

cols_for_widgets = ['type_group', 'corp', 'territory', 'level_national', 'level_regional']

sort_widgets = {}

for col in cols_for_widgets:
    sort_widgets[col] = MultiSelect(title=ENG_TO_RU[col].capitalize(),
                                    value=['All'],
                                    options=['All']+list(df_cap[col].unique()))

widgets_col = column(*sort_widgets.values())
#sizing_mode="fixed", height=250, width=150

def update(attr, old, new):

    filter_dict = {col: sort_widgets[col].value for col in cols_for_widgets if 'All' not in sort_widgets[col].value}

    print("filtered_dict: ", filter_dict)

    if filter_dict:
        df = filter_df(df_cap, filter_dict)
    else:
        df = df_cap

    df_final = make_capital_df(df)

    source_cap.data = df_final # make_capital_df(df_cap, freq='Q', filter_dict=filter_dict)

    filter_dict_full = {col: sort_widgets[col].value for col in cols_for_widgets}
    w_name = [k for k,v in filter_dict_full.items() if v == new][0]

    print("FIND WIDGET: ", w_name)

    idx = cols_for_widgets.index(w_name)

    print("IDX", idx)

    for col in cols_for_widgets[idx+1:]:
        sort_widgets[col].options = ['All']+list(df[col].unique())

for w in sort_widgets.values():
    w.on_change('value', update)

"""
#[END CREATE WIDGETS]

# equipment = column(title_plot,
#                    row(column(p, sizing_mode="stretch_both"), widgets_col),
#                    sizing_mode="stretch_both")

sensor = column(p, select, histplot)

curdoc().add_root(sensor)
curdoc().title = "Sensor analysis"
