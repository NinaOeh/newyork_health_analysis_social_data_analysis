'''
Code for the project website using streamlit.
To view the website, run the code locally on your browser.
'''
import streamlit as st
import pandas as pd
import folium
import json
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import matplotlib

#import functions for the different datasets
import modules.health_functions as health_func
import modules.air_functions as air_func
import modules.tree_functions as tree_func
import modules.model_functions as ml_func
import modules.corr_functions as cr_func

from PIL import Image


#
from matplotlib.backends.backend_agg import RendererAgg
matplotlib.use("agg")
_lock = RendererAgg.lock

#set to wide mode
st. set_page_config(layout="wide")
#remove warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

#load the data
geodata =  json.load(open('./data/geo_files/json/CD.json', 'r'))
geo = json.load(open("./data/geo_files/json/community_districts_updated.geojson"))

@st.cache(allow_output_mutation=True)
def load_health_data():
    health_data = pd.read_csv('./data/health/2015_Health2.csv',sep=";")
    # filtering the data according to required subsets
    health_data.drop(labels=[0,1,2,3,4,5],axis=0,inplace=True)
    health_data=health_data.reset_index()
    focus_health=["ID","Name","Avoidable_Asthma","Poverty","Unemployment","Self_rep_health","Smoking","Life_expectancy_rate"]
    health_data=health_data.loc[:, health_data.columns.isin(focus_health)]
    health_data=health_data.iloc[:59]
    return health_data

@st.cache
def load_tree_data():
    df_trees_1995 = pd.read_csv("./data/trees/trim_final_clean_new_york_tree_census_1995.csv")
    df_trees_2005 = pd.read_csv("./data/trees/trim_final_clean_new_york_tree_census_2005.csv")
    df_trees_2015 = pd.read_csv("./data/trees/trim_final_clean_new_york_tree_census_2015.csv")
    
    return df_trees_1995,df_trees_2005,df_trees_2015

def tree_density_data(geo,df):
    df_counts = pd.read_csv("./data/trees/areas_tree_counts.csv")
    return df_counts

@st.cache(allow_output_mutation=True)
def load_airquality_data():
    df = pd.read_csv("./data/air_quality/Air_Quality.csv")
    df['Start_Date'] =  pd.to_datetime(df['Start_Date'], format='%m/%d/%Y')
    df['Start Year'] = pd.DatetimeIndex(df['Start_Date']).year
    df['Date'] = pd.to_datetime(df['Start_Date'], format='%M/%d/%Y')
    return df

#%%%%%% Map Plot

#Map of NY
def ny_map():
    map_hooray = folium.Map(location=[40.7, -73.86],
                        width="%60",
                        height="%70",
                        zoom_start = 10) # Uses lat then lon. The bigger the zoom number, the closer in you get
    return map_hooray # Calls the map to display

def ny_map_trees(df_trees_2015):
    tree_type_list=df_trees_2015["spc_latin"].unique()
    
    #get ten most common tree-types:
    n = 10
    tree_type_list=df_trees_2015["spc_latin"].value_counts()[:n].index.tolist()
    
    # Filter the DF for rows, then columns, then remove NaNs
    heat_df = df_trees_2015[df_trees_2015["spc_latin"].isin(tree_type_list)] # Reducing data size so it runs faster
    
    
    heat_df = heat_df[['latitude', 'longitude']]
    heat_df = heat_df.dropna(axis=0, subset=['latitude','longitude'])
    
    
    m = folium.Map(
        location=[40.7, -73.86],
        width="%60",
        height="%70",
        zoom_start=10,
    )
    
    # List comprehension to make out list of lists
    heat_data = [[row['latitude'],row['longitude']] for index, row in heat_df.iterrows()]
    
    # Plot it on the map
    HeatMap(heat_data,radius = 5, min_opacity = 3,gradient={0.3:'darkgreen',.6: 'green',0.8:'yellow', .9: 'orange'}).add_to(m)
    
    #def plotDot(point):
    #    '''input: series that contains a numeric named latitude and a numeric named longitude
    #    this function creates a CircleMarker and adds it to your this_map'''
    #    folium.CircleMarker(location=[point.latitude, point.longitude],
    #                        radius=2,
    #                        weight=5).add_to(m)
    
    #use df.apply(,axis=1) to "iterate" through every row in your dataframe
    #heat_df.apply(plotDot, axis = 1)
    
    # Display the map
    return m



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Data load
# Load the data
health_data = load_health_data()
tree_data=load_tree_data()
tree_dens_data=tree_density_data(geo,tree_data)
airquality_data=load_airquality_data()

#load images and assets
correlation_im = Image.open("./data/images/corr.png")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%CSS LAYOUT 

#def local_css(file_name):
#    with open(file_name) as f:
#        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#local_css("modules/style.css")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PAGE SETUP

#set title
st.title('Do trees influence the health of NY citizens?')
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Add the overview of the article to the sidebar:
add_selectbox = st.sidebar.markdown('''
# Sections
- [Introduction](#introduction)
- [Data Exploration](#data_exploration)
- [Correlation Analysis between airquality, health and tree density](#correlation)
- [Machine Learning model](#machinelearning)
- [Predicting health by trees?](#prediction)
- [Conclusion](#conclusion)
''', unsafe_allow_html=False)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
st.header("Introduction")
left_column1, right_column1 = st.columns((2,1))
with left_column1:
    st.markdown("Imagine yourself, sitting on a bench and listening to the wind, \
                that softly strokes and shuffles the soft, green leaves of the tree \
                standing next to the bench. You look up and are blinded by a few \
                beams of afternoon sun cheekily peeking through the umbrella \
                of leaves, forcing you to close your eyes and give in to \
                the moment. There is the song of a blackbird, singing \
                to the scene from one of the branches of the tree \
                and the distinct reply of another blackbird \
                further away. You also hear the buzzing of insects that \
                feed on the trees’ blossoms. Then, suddenly, the loud roar \
                of a combustive engine –  you open your eyes and are back \
                in the city, sitting on one of the few benches next to the main road.")
                    
    st.markdown("We all know the calming effect that nature can have on us and \
                it is proven, that street trees are no exception for that. Scientist \
                that studied the effect of street trees on mental health in France \
                in 2020 found out, that the presence of street trees can even \
                be more efficient than antidepressiva. But what about the \
                physical health? Obviously, due to their carbon-storing \
                nature through photosynthesis, trees should have a positive effect \
                on the air-quality in a city. But is this influence trackable \
                and further more big enough to have an actual impact on the \
                health of the cities inhabitants? Or are social factors more important than city trees?")
    st.markdown("**Let's find out together!** Taking the city of New York as an example\
                we are going explore the health of the citizens, tree distribution and \
                    air pollution and see how and if they are connected.")

with right_column1:
    st.markdown("###### Look at the tree distribution of NY!")
    folium_static(ny_map_trees(tree_data[2]))

# Notify the reader that the data was successfully loaded.
data_load_state.text("")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
st.header("Data Exploration")

st.markdown("For the analysis we used three different data sets. The dataset that we used to obtain the \
            information about the health of the citizens comes from joints data from various American Services\
            It containts information on the health status as well as \
            social status of the citizens of New York City, grouped into the districts of the city. \
            Within this analysis, the used data results from the \
            [New York City Community Health Survey (CHS)](https://www1.nyc.gov/site/doh/data/data-sets/community-health-survey.page) \
            and the [American Community Survey (ACS)](https://www.census.gov/programs-surveys/acs). \
            These surveys are conducted annually by the DOHMH, Division of Epidemiology, \
            Bureau of Epidemiology Services and U.S. Census Bureau. The CHS is a telephone survey, \
            that anually collects random samples of 10000 adults spread across the city of New York. \
            It provides a robust estimation on the health of New Yorkers, including neighborhood, \
            borough, and citywide estimates on a broad range of chronic diseases and behavioral \
            risk factors. The ACS is a nation wide Survey, that focusses on social aspects and \
            wealth, that anually collects data from about 3.5 million addresses across the \
            United States of America.",
            unsafe_allow_html=False)
st.markdown("The tree data come in three files for 1995, 2005, and 2015. \
            It describes each tree as observed by either voluanters or staff, collected by the \
            [New York City Department of Parks & Recreation](https://www.nycgovparks.org/)\
            It includes trees' diameters, heights, and status (health, dead, alive etc)\
            as well as the exact location for each tree, with lang and long, \
            which are also converted to the cartesian plane (X and Y).\
            For our project, we limit our analysis to trees that are NOT dead. \
            For geo analysis, we use the community districts of New York to explore trends. Furthermore, \
            we added park data to the dataset, to represent the green areas of the city. This park data\
            also comes from the [New York City Department of Parks & Recreation](https://www.nycgovparks.org/)\
            in form of a geo-file that states the zipcodes that the park stretches out on and the size on the park\
            in acres",
            unsafe_allow_html=False)
st.markdown("The dataset we used to analyze the airquality comes from the Environment & Health Data\
            Portal, which is run by the NYC Environmental Public Health Tracking Program and contains\
            [New York City air quality surveillance data](http://nyc.gov/health/environmentdata).",
            unsafe_allow_html=False)

        
            
#Options for the reader to look at the datasets
st.success("Please choose one dataset at a time, to explore the data that we used for the analysis!")
left_column2, middle_column2, right_column2 ,very_right_column2= st.columns((1,1,1,1))
with left_column2:
    health_check= st.checkbox('Explore health data')
with middle_column2:
    tree_check=st.checkbox('Explore tree data')
        
with right_column2:
    air_check=st.checkbox('Explore airquality data')
with very_right_column2:
    assumpts_check=st.checkbox('The truth: Assumptions')
        
        
if health_check and not tree_check and not air_check and not assumpts_check:
    left_column_healthdata,middle_column_healthdata, right_column_healthdata ,very_right_column_healthdata= st.columns((7,1,1,1))
    with left_column_healthdata:
        st.markdown("#### Health dataset description")
        st.markdown("The health dataset includes information about social and health factors\
                    of the population, grouped by the 59 community districts of the city. Of the\
                    194 initial variables, 8 are chosen to be included into the analysation of this dataset. \
                    These include the socio-health factors **avoidable asthma**, **unemployment**, \
                    **self-reported health**, **smoking**, and **life expectancy**.")
        with st.expander('See details of health dataset'):
            st.markdown('''
                        Shape of initial dataset:  (68, 194)\
                        Shape of subset of dataset used for analysis: (59, 8)
                        
                        Variables in the dataset:
                        - **ID** (number): Identifier of the neighborhood geographic area, used for joining to mapping geography files to make thematic maps
                        - **Name** (text): Name of the neighborhood geographic area
                        - **Avoidable_Asthma** (number): Age-adjusted avoidable hospitalizations for asthma per 100,000 adults (AHRQ PQIs 5 and 15) (Individuals may report a homeless shelter, a post office box, the Post Office’s General Delivery Window or a residential treatment facility as their residential address. This may lead to higher hospitalization rates in some neighborhoods, depending on what institutions or residential facilities are located in the Community District)
                        - **Poverty** (number): Percent of individuals living below the federal poverty threshold
                        - **Unemployment** (number): Percent of the civilian population 16 years and older that is unemployed
                        
                        - **Self_rep_health** (number): Age-adjusted percent of adults reporting that their health is “excellent,” "very good," or “good” on a 5-level scale (Poor, Fair, Good, Very Good or Excellent)
                        - **Smoking** (number): Identifier of the neighborhood geographic area, used for joining to mapping geography files to make thematic maps
                        - **Life_expectancy_rate** (number): Life expectancy at birth in years 
                        ''')
    
            st.write(health_data)
        st.markdown("Given that the dataset already grouped all individual data into the districts of the city\
                    we can make use of that and look at the distribution of the different factors across the city.\
                    Use the dropdown menu to the right to choose between the different variables.")
        # Create a text element and let the reader know the data is loading.
        data_load_state = st.text('Loading data...')
        st.plotly_chart(health_func.health_data_on_map(health_data,geodata),use_container_width=True)
        # Notify the reader that the data was successfully loaded.
        data_load_state.text("")
        st.markdown("To see if there are possible correlations between the different variables, in a next step\
                    we have a look at the four poorest and four richest districts, and how they are ranked in \
                    relation.")
        st.pyplot(health_func.poverty_ranking(health_data),use_container_width=True)
        st.markdown("This visualization allows us to see possible connections. For example are 3/4 \
                    of the poorest districts (the districts with the highest poverty)\
                    also the districts with the highest unemployment rate - that makes\
                    sense! Furthermore, we can see that the poor districts report to feel less healthy, which can\
                    also be seen in the life-expectancy, that is lower for people in the poorer districts.\
                    Interesting is also, that the asthma rate is higher for the poorer districts. Looking at the \
                    linear correlation in pairplots between the avoidable asthma rate and all other variables and the life \
                    expectancy and all other variables, reinforces the assumption that health and social status \
                    seem highly correlated.")
        st.plotly_chart(health_func.correlation_two_var(health_data),use_container_width=True)
        with st.expander('Look all the pairplots to see the correlation matrix'): 
            st.pyplot(health_func.pairplot_health(health_data),use_container_width=True)
        st.markdown("This dataset shows us that there seems to be a large correlation between the social status\
                    of a person and their health. Possibly this relationship might be more important than the \
                    airquality or tree density? Let's keep that in mind while we continue with the analysis!")
elif tree_check and not health_check and not air_check and not assumpts_check:
    left_column_treedata, middle_column_treedata, right_column_treedata, very_right_column_treedata= st.columns((1,7,1,1))
    with middle_column_treedata:
        st.markdown("#### Tree dataset description")
        st.markdown("The tree dataset includes information about the street trees in the city\
                    of New York. This does not inlcude park trees, but trees that grow alongside roads.\
                    In the dataset there is one row for each tree, showing the position, health,\
                    kind of tree and other tree related information. The available data contains the \
                    tree count from the years 1995, 2005 and 2015. Have a look at the first 50 rows of the dataset\
                    to understand, how the dataset is built up.")
        with st.expander('Show tree dataset'):
            chosen = st.radio(
                'Tree data from',
                ("1995", "2005", "2015"))
            if chosen=="1995":
                st.subheader('Tree data set from 1995')
                st.write(pd.concat([tree_data[0].head(50)]))
            if chosen=="2005":
                st.subheader('Tree data set from 2005')
                st.write(pd.concat([tree_data[1].head(50)]))
            if chosen=="2015":
                st.subheader('Tree data set from 2015')
                st.write(pd.concat([tree_data[2].head(50)]))
        st.markdown("To get a better feeling on how the trees are distributed across the city, and to be \
                    able to later compare this data to the health and airpollution dataset, we have grouped the trees according to \
                    the different districts.")
        # Create a text element and let the reader know the data is loading.
        data_load_state = st.text('Loading data...')
        st.plotly_chart(tree_func.tree_count(tree_dens_data,geodata),use_container_width=True)
        # Notify the reader that the data was successfully loaded.
        data_load_state.text("")
        st.markdown("One can see that the amount of trees had changed over the years. Let's look\
                    at the more recent change a bit more in detail.")
        st.plotly_chart(tree_func.tree_change(tree_data, geodata),use_container_width=True)
        st.markdown("Great, it looks like between 2005 and 2015 the NY city council planted\
                    more trees across the city! But there are also several red areas, that \
                    indicate a loss of trees. Eventhough we will not analyse temporal patterns \
                    in this analysis, this is still interesting to have in mind.")
        with st.expander('Check out how much the tree density exactly changed per district'):
            st.pyplot(tree_func.tree_change_per_district(tree_dens_data))
            st.markdown("Only 7 districts have experienced a decrease in tree density from 2005 \
                        to 2015. 4 of which are located in Queens, 2 in Brooklyn, and 1 in Bronx.\
                        Note however that by this metrics, it is more difficult to increase the \
                        trees/km2 for large areas: more trees need to be planted to have a larger \
                        number in bigger areas. Nonetheless, the tree count per unit area is a \
                        standard measure in environmental science and it is pretty descriptive, \
                        hence why it is used here.")
elif air_check and not tree_check and not health_check and not assumpts_check:
    
    left_column_treedata, middle_column_treedata, right_column_treedata, very_right_column_treedata = st.columns((1,1,7,1))
    with right_column_treedata:
        st.markdown("#### Air quality dataset description")
        st.markdown("The air quality dataset includes information about multiple pollutants across the city\
                    of New York from 2006 to 2016. Air pollution is one of the most important \
                    environmental threats to urban populations and while all people are exposed, \
                    pollutant emissions, levels of exposure, and population vulnerability vary \
                    across neighborhoods. Exposures to common air pollutants have been linked \
                    to respiratory and cardiovascular diseases, cancers, and premature deaths. \
                    These indicators provide a perspective across time and NYC geographies to \
                    better characterize air quality and health in NYC.")
        with st.expander('Show airquality dataset'):
                st.markdown('''
                            Variables in the dataset:
                            - **unique_id** (text): Unique record identifier
                            - **indicator_id** (number): Identifier of the type of measured value across time and space
                            - **name** (text): Name of the indicator
                            - **measure** (text): How the indicator is measured
                            - **measure_info** (text): Information (such as units) about the measure
                            - **geo_type_name** (text): Geography type; UHF' stands for United Hospital Fund neighborhoods; For instance, Citywide, Borough, and Community Districts are different geography types
                            - **geo_join_id** (text): Identifier of the neighborhood geographic area, used for joining to mapping geography files to make thematic maps
                            - **geo_place_name** (text): Neighborhood name
                            - **time_period** (text): Description of the time that the data applies to ; Could be a year, range of years, or season for example
                            - **start_date** (floating_timestamp): Date value for the start of the time_period; Always a date value; could be useful for plotting a time series
                            - **data_value** (number): The actual data value for this indicator, measure, place, and time
                            - **message** (text): notes that apply to the data value; For example, if an estimate is based on small numbers we will detail here. In this case they are all NaN
                                                        ''')
                st.write(airquality_data)
        st.markdown("Because of the large amout of data, at first a check was made if there is an \
                    equal amount of data in all districts. The analysis shows that the data is not \
                    equally distributed. The reason for that is that for some years data has not \
                    been collected in some of the districts.")
        # Create a text element and let the reader know the data is loading.
        with st.expander('Data distribution'):
            data_load_state = st.text('Loading data...')
            st.pyplot(air_func.plot_air_equal(airquality_data))
            # Notify the reader that the data was successfully loaded.
            data_load_state.text("")
        st.markdown("Next, we want to look at the distribution of fine particle matter and sulfur\
                    dioxide concentration across the city, \
                    as this will be the data that we will compare with the other datasets.")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        air_viz1=st.radio(label = 'Choose data to visualize geographically', options = ['Fine Particle Matter','Sulfur Dioxide'])
        data_load_state = st.text('Loading data...')
        if air_viz1=='Sulfur Dioxide':
            st.plotly_chart(air_func.sulfur_on_map(airquality_data, geodata),use_container_width=True)
        else:
            st.plotly_chart(air_func.fine_particle_matter_on_map(airquality_data, geodata),use_container_width=True)
        data_load_state.text("")
        st.markdown("It seems like there is quite a big difference between the different \
                    districts of the city. It will be interesting to see, if this is connected to \
                    the tree density and if it has an influence on the health of the population!")
        st.markdown("Finally we want to have a look at the development of the two parameters \
                    over time in each district.")
        air_viz2=st.radio(label = 'Choose data for temporal visualization', options = ['Fine Particle Matter','Sulfur Dioxide'])
        data_load_state = st.text('Loading data...')
        if air_viz2=='Sulfur Dioxide':
            st.bokeh_chart(air_func.sulfur_over_time(airquality_data),use_container_width=True)
        else:
            st.bokeh_chart(air_func.fine_particle_over_time(airquality_data),use_container_width=True)
        data_load_state.text("")
        st.markdown("Well, that's good news! The airpollution has gone down over time in all \
                    districts of the city.")
                    
elif assumpts_check and not tree_check and not health_check and not air_check:
    left_column_assumpdata, middle_column_assumpdata, right_column_assumpdata, very_right_column_assumpdata = st.columns((1,1,1,7))
    with very_right_column_assumpdata:
        st.markdown("#### Assumptions or 'the truth'")
        st.markdown("However nice this might sound so far: we just look at some NYC data, create some code \
                    and visualizations and - Abracadabra - we know if health is influenced by trees and parks\
                    in the city. The truth is that as every statistical and machine learning model, this analysis is just a perception\
                    of the real world, because the real world is a very complex place and it \
                    is impossible to model it without any assumptions. Thus, in order to model it and \
                    trying to make predictions, one needs to simplify and capture the problem with numbers \
                    and assumptions. The numbers that our analysis is \
                    based on can be found and explored in the datasets, so here it's time to get down to \
                    reality and outline the main assumptions and limitations of our analysis.")
        st.markdown('''
                    1.) The available data was collected at different years. While the tree and airquality\
                        dataset are from 2015, the health data was collected between 2011 and 2013 and then \
                        averaged. Thus, our first assumption is that **the health and social class\
                        of the people of NYC has not changed significantly between the different districts \
                        of the city between 2013 and 2015**. 
                        
                    2.) To calculate the tree density and tree count all trees are treated equally and no difference\
                        is made between healthy or ill, big or small trees. Our second assumption is that **\
                        all trees have the same influence.**
                        
                    3.) For the park data, parks that stretched over more than one community district were \
                        distributed equally to these districts, assuming that **big parks distribute equally\
                        across all districts they are part of.** Furthermore, we simplified that **all parks are the same**. As we did not differentiate between playgrounds, running fields and \
                        forests.
                        
                    4.) Finally we assume that **every district is homogenous across itself**. \
                        Thus it is sufficient to group the data into the different communal districts for the analysis.
                    ''')
        



#%%%%%%%%%%%%%%%%%%%%%%% Correlation Analysis 
st.header("Correlation Analysis between airquality, health and tree density")

            
lc, rc = st.columns((1,1))

with lc:
    st.markdown("Now that we know a bit about the data and the assumptions that we \
                made for the analysis, we want to move on to the more interesting part that\
                comes closer to hopefully answering our question, if we will actually\
                be able to see an effect of trees and parks in the city on the physical health of the citizens\
                - the correlation between the datasets.")
    st.markdown("To analyze the correlation between the different datasets, we defined important variables from each\
                dataset, that were relevant for our analysis.\
                Then we merged them together into one dataset, grouped by the unique community districts. \
                ")
    st.markdown('''
                The correlation matrix shows which variables are correlated and which not,\
                claculated with the Pearson coefficient, a linear measure\
                of the correlation  between each of the variables. Big blue boxes symbolize a high positive correlation,
                meaning that when one of the pairs' values increase, the other pairs' values increase too. And big\
                red boxes symbolize a high negative correlation, meaning that when one of the pairs' values increase, \
                the other pairs' values decrease.''')


with rc:
    
    st.pyplot(cr_func.corrplot(airquality_data, health_data, tree_dens_data))
    with st.expander("See the (simple) math behind it"):
        st.markdown("The Pearson correlation coefficient is calculated in the following way:")
        
        st.image(correlation_im)
        st.markdown('''
                    With n being the sample size (in our case the number of districts), \
                    $x_i,y_i$ the individual sample points (in our case the value of the different \
                    parameters in our data for each district). 
                        
                    You can find more information [here](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
                    ''')
        



#%%%%%%%%%%%%%%%%%%%%%%% Machine Learning
st.header("Machine Learning model")

st.write("In this notebook, we create two models that uses data related to the various \
districts of New York to predict the Health related rates among the population.")
st.write("The data related to each district used in the training consists of the following:  \n"\
"""    
* Tree counts  \n \
* District areas  \n \
* Trees' density  \n \
* Percentage district area occupied by parks  \n \
* Fine Particulate Matter (PM2.5) (mcg per cubic meter)  \n \
* Sulfur Dioxide (SO2) (ppb)  \n \
* Poverty rate  \n \
* Smoking rate  \n \
* Unemployment rate  \n \
* Self reported health status
""")
st.write("The models will try to predict the following variables:  \n"\
"""    
* Number of people with Avoidable Asthma per 100 000 people  \n \
* Life expectancy \n \
""")


# Loading the df used for the models, preprocessing included
df_final = ml_func.loading_df() # final df used to build the model

with st.expander("Model creation and predictions"):
	st.write("The two models are created using xgboost. The columns used for training the model are \
          those indicated above, while the target variable are the Avoidable Asthma Rate and the \
        Life Expactancy Rate among the various district for the first and second model respectively. \
        The comparison between the real data and the predicted data is presented below. ")
	# Creation of model 1
	xgb_model_1, y_test_1, y_pred_1, mse_1 = ml_func.training_model_1(df_final)
	# Creation of model 2
	xgb_model_2, y_test_2, y_pred_2, mse_2 = ml_func.training_model_2(df_final)
	
	st.bokeh_chart(ml_func.plot_model_results(df_final, xgb_model_1, y_test_1, y_pred_1, mse_1, xgb_model_2, y_test_2, y_pred_2, mse_2),use_container_width=True)
	
	st.write("Results: As we can see from the graph, the model manages to predict the general \
         behaviour of the health rates analyzed.  \n")

with st.expander("Model Interpretation"):
	st.write("We want now to analyze the importance of each variable used to predict the two health \
         factors. Which one is more important? Tree density or Poverty rate, Smoking rate or \
            Sulfur Dioxide levels in the district?")
	# plot model interpretability
	st.pyplot(ml_func.model_interpretation(df_final, xgb_model_1, xgb_model_2),use_container_width=True)
	st.write("As we can see from the graph above, the most important features calculated from the \
        prediction model are the social aspects of the population. Unemployment and Poverty \
        appear to be the most important variables in avoiding Asthma and extending the Life Expectancy. \
        The latter, appears to be also determined by the Fine particular matter (PM2.5) level in \
        the districts. Trees density appear to have very low effect in predicting the health of \
        the population.")
	
#%%%%%%%%%%%%%%%%%%%%%%%Conclusion
st.header("Conclusion")
st.write("The purpose of this analysis was to understand what is the relation between health, \
trees and air quality among the districts of New York City. We studied different datasets, \
and built a Machine Learning Model that predicted Avoidable Asthma Rate and Life Expectancy \
Rate in 2015 using social aspects, air quality levels and tree density in the various districts. \
The results are that the most important variables when considering health are social factors \
related to the economic conditions of the people studied, in our case employment and poverty. \
Air pollution also affects the Life expectancy, although we only found low influence when \
considering the Sulfur Dioxide levels, probably because of the low variance between district \
in the chosen year. On te other hand, the amounth of trees in a districts appears to have very \
low importance. The reason behind this is that probably the effect of trees on health can only \
be seen in a greater scale, and not analyzing each specific district.")

#%%%% Hide Menu
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)