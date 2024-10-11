#!/usr/bin/env python
# coding: utf-8

# 
# # Introduction
# 
# In this series of excercises, you will learn how to work with ase databases,
# and do some simple machine learning for electronic structure properties. The
# driving idea is to predict complex properties of compounds from simpler
# properties, under the slogan that the fastest calculation is the one you
# don't have to run. We start by importing some relevant packages for
# scientific python and `ase` in particular.
# 

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect
from ase.visualize import view


# 
# In the current directory, there is an `ase` database file called 'organometal.db'.
# It contains information about organometallic perovskites, and the goal is to
# predict properties for these. Along with the perovskite compounds, there are
# also reference calculations of the elements in their standard states. We
# start by connecting to the database (more info on the `ase db` module can be
# found [here](https://wiki.fysik.dtu.dk/ase/ase/db/db.html#module-ase.db)),
# and inspecting a single row:
# 

# In[3]:


db = connect('organometal.db')
row = next(db.select(project='organometal'))
vars(row)


# 
# Each row of the database has some key-value pairs, which are stored explicitly, as well as some basic information which is always stored, recording how the data was calculated. Can you identify what each of the keys below refers to?
# 
# calculator:
# calculator_parameters:
# cell:
# ctime:
# energy:
# gllbsc_dir_gap:
# gllbsc_disc:
# gllbsc_ind_gap:
# id:
# initial_magmoms:
# mtime:
# name:
# numbers:
# pbc:
# positions:
# project:
# space_group:
# symmetry:
# unique_id:
# user:
# 

# 
# Each row also has a `toatoms()` method, which lets us extract an `ase.Atoms` object from the row.
# 

# In[4]:


row.name


# In[5]:


row.symbols


# In[5]:


from IPython.display import Image
Image(filename='ABX3_perovskite.jpeg',width=400) # ACS Energy Lett. 5, 604 (2020), doi: 10.1021/acsenergylett.0c00039


# 
# When doing any kind of data analysis, the first step is always to become familiar with the data in question, on a basic level. The `select()` method of the database applies a query to the database and returns an iterator of all the rows matching that query. To select all rows with a user of `einstein`, we would type `db.select(user='einstein')`. To select all rows with a gllbsc direct gap greater than 0.5 eV, we would type `db.select('gllbsc_dir_gap>0.5')`.
# Counting the number of hits can be be done using `db.count(key=value)` for some key-value pair.
# 
# How many rows are there in the database?
# How many belong to the `organometal` _project_? And how many to the `references` _subproject_?
# 

# In[6]:


db.count()  # number of rows in the database


# In[7]:


db.count(project='organometal')  # number of rows belong to the organometal project


# In[8]:


db.count(subproject='references')  # number of rows for references subproject


# 
# The structures in the database were generated from the general formula ABX,
# and then varying A, B and X. X represents a combination of 3 halogen atoms,
# chosen from ["Cl", "Br", "I"]. The A, B and X is encoded in value for the key
# `name`, i.e. `row.name -> 'CsPbI3'`. We have also distorted some of the
# structures, giving four different symmetry types for each atomic composition.
# 
# 1. Try to identity the possible values of A and B.
#    (Hint: A and B is labeled with two characters `row.name`,
#    i.e `A='Cs'` and `B='Pb'` in `'CsPbI3'`)
# 
# 2. Can you identify the four different symmetry classes?
# 
# 3. By making all possible combinations of both A, B, X, and symmetries, how
#    many structures could be generated in total? And how many unique are there,
#    i.e. without considering the different symmetries?
# 

# In[9]:


names=[]
syms=[]

for row in db.select(project='organometal'):
    names.append(row.name)
    syms.append(row.symmetry)
label_A=[name[0:2] for name in names]
label_B=[name[2:4] for name in names]
label_X3=[name[4:] for name in names]


# In[10]:


import pandas as pd

def echo_unique(slist):
    ulist=pd.Series(slist).drop_duplicates().tolist()
    print(ulist)
    return ulist


# In[11]:


echo_unique(label_A)
echo_unique(label_B)
echo_unique(label_X3)


# In[12]:


unique_sym=echo_unique(syms)


# 
# As you can see from the exercise above, two organic molecules (methylammonium MA, formula CH$_6$N and formamidinium FA, formula CH$_5$N$_2$) can be used instead of Cs as cations in the inorganic perovskite template. View the structure of MA and FA from the reference subproject in the database.
# 

# In[13]:


for row in db.select(subproject='references'):
    if row.formula =='CH5N2':
        view(row.toatoms())


# 
# Two good ideas are to plot distributions of interesting quantities, and to calculate some summary statistics of numeric quantities, such as the mean and the variance.
# 

# In[14]:


organometal_rows = [x for x in db.select(project='organometal')]
discs = [x.get('gllbsc_disc') for x in organometal_rows]
print(f"Mean of `gllbsc_disc` = {np.mean(discs):.4f}")
print(f"Variance of `gllbsc_disc` = {np.var(discs):.4f}")
plt.hist(discs)
plt.xlabel('Energy (eV)')
plt.show()


# In[15]:


x=range(db.count(project='organometal'))
plt.scatter(x,discs)


# 
# Make a histogram for each of the numeric quantities in the database, and calculate the mean and variance of these quantities. You can also make scatter plots of one quantitity against another by using the `scatter` method of pyplot, `plt.scatter()`. How do these distributions vary with the `symmetry` keyword of the database?
# 

# In[16]:


numeric_quantities = ["energy", "gllbsc_disc", "gllbsc_dir_gap", "gllbsc_ind_gap"]
om_rows = [x for x in db.select(project='organometal')]
results=[]
fig, axs = plt.subplots(2, 2)

for prop in numeric_quantities:
    values=[row.get(prop) for row in om_rows]
    results.append(values)
    print(f"Mean of `{prop}' = {np.mean(values):.4f}")
    print(f"Variance of `{prop}' = {np.var(values):.4f}")

axs[0,0].hist(results[0])
axs[0,0].set_title(numeric_quantities[0])
axs[0,1].hist(results[1])
axs[0,1].set_title(numeric_quantities[1])
axs[1,0].hist(results[2])
axs[1,0].set_title(numeric_quantities[2])
axs[1,1].hist(results[3])
axs[1,1].set_title(numeric_quantities[3])
fig.tight_layout(pad=1.5)


# In[19]:


numeric_quantities = ["energy", "gllbsc_disc", "gllbsc_dir_gap", "gllbsc_ind_gap"]
om_rows = [x for x in db.select(project='organometal',symmetry=unique_sym[3])]  # cubic samples
results=[]
fig, axs = plt.subplots(2, 2)

for prop in numeric_quantities:
    values=[row.get(prop) for row in om_rows]
    results.append(values)
    print(f"Mean of `{prop}' = {np.mean(values):.4f}")
    print(f"Variance of `{prop}' = {np.var(values):.4f}")

axs[0,0].hist(results[0])
axs[0,0].set_title(numeric_quantities[0])
axs[0,1].hist(results[1])
axs[0,1].set_title(numeric_quantities[1])
axs[1,0].hist(results[2])
axs[1,0].set_title(numeric_quantities[2])
axs[1,1].hist(results[3])
axs[1,1].set_title(numeric_quantities[3])
fig.tight_layout(pad=1.5)


# 
# The energy contained in each row is an energy with respect to some arbitrary reference, which was set in the original calculation. A more sensible reference is provided by the materials with `subproject == 'references'`. We can calculate the heat of formation per unit cell of the 'MAPbI3' compound as follows:
# 

# In[20]:


row = db.get(name='MAPbI3', symmetry='cubic')
en_cubic = row.energy
en_refs = {}
for row in db.select(subproject='references'):
    en_refs[row.element] = row.energy / row.natoms

E_standard = en_cubic - (8 * en_refs['MA'] + en_refs['Pb'] + 3 * en_refs['I'])
print(f'hof = {E_standard / row.natoms:.3f} eV/atom')


# 
# Based on this, can you calculate the difference in heat of formation per formula unit of MAPbI$_3$ in the tetragonal phase versus the cubic phase? What about the heat of formation of FASnBr$_2$Cl in the orthorhombic_1 phase versus the cubic phase of the FASnBr$_3$, and of FASnCl$_3$. (hint: tetragonal and orthorhombic phases have a unit cell larger than the cubic structure).
# 

# 
# # Machine Learning
# 
# Machine Learning is the science of getting computers to learn and act like humans do, and improve their learning over time in autonomous fashion, by feeding them data and information in the form of observations and real-world interactions. The crucial idea is that the computer should be able to improve its performance at a given task as we give it more information. A tutorial on machine learning in general can be found [here](https://www.tutorialspoint.com/scikit_learn/index.htm).
# 
# In this workbook we will be carrying  out a supervised learning task, where we attempt to predict a particular (known) attribute of a given structure, based on other attributes of the structure. This can be useful if it allows us to **use easy-to-calculate properties to say something about quantities which are difficult to calculate**. This approach to learning, where we attempt to find a map $f$ from the attributes, $X$, of our data to some target property, $y$, is known as supervised learning. See [here](https://www.datacamp.com/blog/supervised-machine-learning) for more general information on the topic.
# 
# The two most important ingredients in the supervised learning approach are which attributes of the data we feed into the machine learning process, and which model we then apply on the data.
# 
# ## Input Vectors
# 
# To start, we use a one-hot encoding of each of the different categories of data as our input vector. Later, you will be asked to see if you can find a better set of features to describe the data.
# 

# In[35]:


def calculate_input_vector(row):
    symm_vec = [0, 0, 0, 0]
    A_vec = [0, 0, 0]
    B_vec = [0, 0]
    X_vec = [0, 0, 0]  # i.e I3->[0, 3, 0], I2Cl->[1, 2, 0], Br3->[0, 0, 3]
    constant = [1,]
    symm_vec[['cubic',
              'tetragonal',
              'orthorhombic_1',
              'orthorhombic_2'].index(row.symmetry)] = 1
    A_vec[['Cs', 'FA', 'MA'].index(row.name[:2])] = 1
    B_vec[0] = 1 if 'Pb' in row.name else 0
    B_vec[1] = 1 if 'Sn' in row.name else 0

    Xs = ['Cl', 'I', 'Br']
    nhalo = sum([s in Xs for s in row.symbols])
    for i, X in enumerate(Xs):
        X_vec[i] = 3 * sum([s == X for s in row.symbols]) // nhalo

    vec = symm_vec + A_vec + B_vec + X_vec + constant
    return vec


# 
# In a one-hot encoding, assign the data into different categorical classes, and then have one feature for each class. For example, there are four different symmetries in the data, so the first four features of the input vector describe which symmetry the material lies in. A '1' in a given position indicates that the material falls into that class, while a '0' indicates that it does not.
# 
# As an example, we apply the encoding to the first few rows of the database with cubic symmetry, and show the formula and the name:
# 

# In[36]:


for row in db.select(symmetry='cubic', limit=5):
    print(f'name={row.name} formula={row.formula} symmetry={row.symmetry}')
    print(f'vec={calculate_input_vector(row)}')
    print('-'*79)


# 
# We see that as expected, the row has a '1' in the first position, indicating that it has a cubic symmetry.
# 
# We are now ready to generate the input matrix $X$ that we will use in the machine learning process. At the same time we extract the indirect band gap as the target vector, $y$.
# 

# In[37]:


X = []
y = []
for row in db.select('project'):
    X.append(calculate_input_vector(row))
    y.append(row.gllbsc_ind_gap)

X = np.array(X)
y = np.array(y).reshape(-1, 1)
print('X.shape = ', np.shape(X))
print('Y.shape =', np.shape(y))


# 
# ## Modelling
# 
# With the input and output in place, we are ready to do the initial machine
# learning. All supervised machine learning processes do the following, in a
# generalized sense:
# 
# - Select a general functional form for the model, parametrized in
#   a suitable way.
# 
# - Find a loss function to evaluate the performance of a given set
#   of parameters.
# 
# - Optimize the parameters of the model to minimize the loss.
# 
# All the hard work is usually in the last step, since for complex models the
# relationship between the parameters and the loss function can be very
# difficult. However, for simpler models, we can sometimes find a closed form
# for the loss function.
# 
# The very simplest class of machine learning models are just generalized
# linear models, where the target, $y$, is assumed to be a linear function of
# the input variables. You can read more about them
# [here](http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares). For a
# linear function, we guess a functional form $f(\mathbf{x}) = \sum_n w_n x_n = \mathbf w \cdot \mathbf x$, and seek to optimize the weight vector, $\mathbf w$.
# 
# If we choose the loss function
# $$ L = \sum_i (f(\mathbf{x}_i) - y_i))^2 = \sum_{i} \left(\sum_{n}w_i x_{in} - y_i\right)^2 $$
# , we will recover ordinary least squares regression. In matrix terms, the loss corresponds to the norm $ L = \left\| \mathbf{y} - \mathbf{X} \mathbf{w} \right\|^2$.
# The loss is minimal when the derivative with respect to the weight vector is zero. A bit of
# rearranging gives that this is true when
# $$ \mathbf w = (\mathbf{X}^T\mathbf{X}) ^ {-1} \mathbf{X}^T \mathbf{y} $$
# 
# Here $\mathbf w$ is an (n_features, 1) weight vector that we are trying to
# find, $\mathbf{X}$ is an (n_samples , n_features) matrix of our observed
# inputs and $\mathbf y$ is the (n_samples, 1) output vector that we are trying
# to predict.
# 
# Write a function `fit`, which takes as input a matrix $\mathbf X$, and a
# target vector $\mathbf y$, and performs this linear regression, returning the
# list of weights and an estimate of the loss for this list of weights.
# 
# Hint: useful functions are `np.dot()` and `np.linalg.inv()`, which calculate
# the dot product and inverse, respectively, of their arguments.
# 

# In[ ]:


def fit(X, y):
    """
    code goes here
    """


# In[ ]:





# 
# You can test your code on the following toy input - hopefully the weight vector you find is close to [1, 2, 3, 4, 5]
# 

# In[ ]:


nsamples = 50
nfeatures = 5
X_toy = np.random.rand(250).reshape(nsamples, nfeatures)
coefficients = np.arange(1, nfeatures + 1).reshape(-1, 1)
noise = np.random.normal(scale=0.2, size=nsamples).reshape(-1, 1)
y_toy = np.dot(X_toy, coefficients) + noise
w, loss = fit(X_toy, y_toy)
plt.scatter(np.dot(X_toy, w), y_toy)
plt.plot(plt.xlim(), plt.ylim(), ls='--', color='k')

plt.show()
print(w)


# 
# Once that is working, try it on the original data! (hint: you have already calculated the matrix $X$ and target vector $y$ above) Does everything work as it should?
# 

# In[ ]:





# 
# One of the assumptions made in the derivation of the linear model is that the matrix $(\mathbf{X}^T\mathbf{X})$ is invertible. Unfortunately, that's not true for our case. That's because of the encoding we have chosen, which means for example that for any row, the first four columns must sum to one. The fourth column can therefore always be written as 1 - the sum of the first three.
# 
# We can alleviate this by adding a regularization term to our loss function, which penalises large weights. The new loss is can then be written as  $L = \left\| \mathbf{y} - \mathbf{X} \mathbf{w} \right\|^2 + \alpha \left\| w\right\|^2$. Luckily, there is still a closed-form solution for this, namely $\mathbf w = (\mathbf{X}^T\mathbf{X} + \alpha \mathbf{I}) ^ {-1} \mathbf{X}^T \mathbf{y}$. Modify your `fit()` function to take an extra argument $\alpha$, and apply regularization to the original problem. Does everything work now? Do the weights of your model make sense? Try few different values for $\alpha$. How does the fit changes with $\alpha$?
# 

# In[ ]:





# 
# We can do all the things above in scikit-learn, which is a python software package for doing many different kinds of modelling.
# 

# In[ ]:


from sklearn import linear_model
linear = linear_model.LinearRegression()

linear.fit(X, y)
ybar = linear.predict(X)
ymax = np.array((y, ybar)).max() + 0.1
plt.scatter(ybar, y)
lims = [
    np.min([plt.xlim(), plt.ylim()]),
    np.max([plt.xlim(), plt.ylim()])
]
plt.plot(lims, lims, ls='--', color='k')
plt.xlabel('Predicted Band Gap [eV]')
plt.ylabel('Actual Band Gap [eV]')

# We can wrap the above in a function, to avoid typing that same code again later
def limits():
    lims = [
        np.min([plt.xlim(), plt.ylim()]),
        np.max([plt.xlim(), plt.ylim()])
    ]
    plt.plot(lims, lims, ls='--', color='k')

def make_comparison_plot(X, y, model, label):
    model.fit(X, y)
    ybar = model.predict(X)
    plt.scatter(ybar, y)
    limits()
    plt.xlabel(f'Predicted {label}')
    plt.ylabel(f'Actual {label}')


# 
# All the models in scikit-learn have a `fit` method, which expects an $X$ matrix and a $y$ vector as inputs, and then trains the model. They also have a `predict` method, which takes an $X$ matrix as input and returns the $y$ values predicted by the model. We use this to plot the true vs the predicted band gap.
# 

# 
# We can also inspect the parameters of the model to see which elements of the input vector are important to the model:
# 

# In[ ]:


print(linear.coef_)
print(linear.intercept_)


# 
# We see that despite the singular matrix, sklearn is able to do a linear fit, and returns sensible coefficients. Relying on this behaviour is rather brittle, and a better idea is to do as before and add a regularization parameter to the original loss function, which is done in the `linear_model.Ridge` model:
# 

# In[ ]:


from sklearn import linear_model

linear_regularized = linear_model.Ridge(alpha = .5)
make_comparison_plot(X, y, linear_regularized, "Band Gap [eV]")


# In[ ]:


print(linear_regularized.coef_)
print(linear_regularized.intercept_)


# 
# From visual inspection, it seems that the regularized model performs about as well as the original linear model.
# 
# To proceed with the machine learning, we need some way of evaluating the performance of a model, which is better than visual inspection of predicted versus actual values, and an assessment of the reasonableness of model parameters. Scikit-learn provides a `score()` method for each model, which evaluates how good the fit is.
# 

# In[ ]:


linear_regularized.score(X, y)


# 
# To truly compare between models, we should ideally train on some data, and evaluate the model on a different set of data. Otherwise, we could create a perfect model just by storing all the input data, and looking up the correct answer. The way to do this is by cross-validation, where the data is randomly split into a number of buckets, and for each bucket, the model is trained on all the other data, and then tested on the data in the bucket. Since the data might have a meaningful order in the database, it is important that the assignment of the data to each bucket is done at random. This is accomplished by the `shuffle` argument to `KFold`.
# 
# This approach of evaluating the performance of a model is very general and can also be used to optimize the so-called hyperparameters of a model, such as the regularization parameter $\alpha$. Here, we will not optimize $\alpha$, but only compare the performance of the `alpha=0` and `alpha=0.5` model. The score has been chosen so that the closer it is to 1, the better.
# 

# In[ ]:


from sklearn import model_selection
folds = model_selection.KFold(n_splits=2, shuffle=True, random_state=1)
print(model_selection.cross_val_score(linear_regularized, X, y, cv=folds, scoring='explained_variance'))
print(model_selection.cross_val_score(linear, X, y, cv=folds, scoring='explained_variance'))


# 
# ## Modelling the heat of formation
# 
# Having looked at the band gap, we turn now to the heat of formation, which was defined further up. Try using the heat of formation as a target vector $\mathbf y$ instead of the band gap. See if it is possible to predict the heat of formation using (regularized) linear regression and the simple input vector defined above. You can use the following code to calculate the heat of formation of all the compounds in the database. Can you explain what it does?
# 

# In[ ]:


en_refs = {}
for row in db.select(subproject='references'):
    en_refs[row.element] = row.energy/len(row.symbols)
HoF = []
for row in db.select(project='organometal'):
        energy = row.energy
        # how many units are in there!?
        n_units = len([True for symbol in row.symbols if symbol == 'Pb' or symbol == 'Sn'])
        energy_standard = 0
        for symbol in row.symbols:
            if symbol in ['Cs','Pb','Sn','I','Br','Cl']:
                energy_standard += en_refs[symbol]
        if 'FA' in row.name:
            energy_standard += n_units * en_refs['FA'] * 8
        if 'MA' in row.name:
            energy_standard += n_units * en_refs['MA'] * 8
        HoF.append((energy-energy_standard) / n_units)


# In[ ]:





# 
# When searching for new materials, we would like to find only stable materials. These should have a negative heat of formation, and be in the most stable symmetry class of the four we are considering. The model we have just made will tell us whether a given compound has a negative heat of formation, but it won't tell us whether a different symmetry class would have a lower energy.
# 
# Predicting which symmetry class of the four is most stable for a given composition is a typical example of a classification problem: we would like to map a given composition, to which of the four symmetry classes is most stable. We start by creating the new output vector corresponding to our data. The output vector should have dimensions (n_compositions, 1), where n_compositions is the total number of different possible compositions without considering symmetries. For each composition, the output should indicate which of the symmetries in the database gives the lowest energy. An idea could be to reuse the `symmetry map` from further up.
# 
# For the input vector, we can use the same one as before, only with the first four entries (corresponding to the symmetry class) removed.
# 
# You should start by creating the target vector, which describes which symmetry is most stable for each composition:
# 

# In[ ]:


# Define `X_hof` and `y_hof` in here


# 
# Once you have that, we can start modelling this data. To try something different, we will be using a decision tree to classify the data. This can be imported as follows:
# 

# In[ ]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(?, ?)


# 
# We can visualize the tree using `graphviz`. Can you explain it?
# 

# In[ ]:


import graphviz
feature_names = ['first_cs','first_FA','first_MA','lead','tin','chlorine','iodine','bromine','reg']
target_names = ['cubic', 'tetragonal', 'orthorhombic_1', 'orthorhombic_2']
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names = feature_names,
                                class_names = target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph


# 
# One issue with decision trees is that it's often easy to overfit the data, by making the depth of the tree too large. To see if this is occurring, we can compute the cross-validation score of the model for different sets of maximum depths:
# 

# In[ ]:


folds = model_selection.KFold(n_splits=4, shuffle=True, random_state=0)
scores = []
for max_depth in range(1, 10):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    score = model_selection.cross_val_score(clf, X_hof, y_hof, cv=folds)
    print(max_depth, score)
    scores.append(score.mean())
plt.plot(range(1, 10), scores)
plt.xlabel('Maximum depth')
plt.ylabel('Cross-validation score')
plt.show()


# 
# ## Improving the input vector
# 
# There is much more information we could include in the input vector, which might increase our predictive power of our model. An example could be the covalent radius of tin vs lead, or of the halogens, which will tend to increase or decrease the lattice constant of the whole crystal, and hence the energy. The covalent radii are available in `ase.data`, and can be imported as `from ase.data import covalent_radii`. Another example could be the electronegativities of the halogens, which will play a role in the character of the bonds. The mulliken electronegativities of chlorine, bromine and iodine are 8.3, 7.59 and 6.76 respectively
# 
# Start by redoing a basic examination of the data: plot the band gap and heats of formation against some of the quantities you might add, and see if there is any correlation. For the quantities that do have a correlation, try adding them to the input vector, and see which (if any) result in an improved model for either the heat of formation, the relative energetic ordering of the symmetry classes, or the band gap.
# 

# In[ ]:





# In[ ]:





# In[ ]:





# 
# ## Improving the models
# 
# Most phenomena we encounter don't have nice linear relationships between inputs and outputs. We model nonlinear relationships in two different ways: we either introduce nonlinearity in our features, or in our model. The first
# is known as feature engineering, and requires a good method. The second makes optimizing the parameters of our model slightly more difficult, as we can lose many of the nice properties of the linear model, such as the closed-form solution.
# 
# Here we will focus on gaussian process regression as a case study. You can read more about it [here](http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process). This lets us fit nonlinear functions and also provides a confidence interval indicating how well the machine is doing.
# 
# Similar to the `linear_regression`, we can import it from `sklearn` and create an example of the model with `model = GaussianProcessRegressor(kernel=YYY)`, where `YYY` is the kernel used. As a start, we should use a radial basis function, which is also available from `sklearn`. As usual, the model has a `fit()` and a `predict()` method, which sets the parameters of the model, and tries to predict outputs based on inputs.
# 

# In[ ]:


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = RBF(length_scale=0.1)
model = GaussianProcessRegressor(kernel=kernel)
model.fit(X, y)
fit = model.predict(X)
model.score(X, y)


# 
# The model we have chosen is very dependent on the length scale we set for the kernel, and it is not given that the value chosen above (0.1) is a good choice. The fact that we have a score of 1.0 is an indication that we may be overfitting. The trick to selecting the best value of this parameter is again cross-validation. We can loop over different possible values of the hyperparameter to make different classes of models, and then evaluate the cross-validation score of each to see which performs best. Try this!
# 

# In[ ]:





# 
# Ideally you should find that length scales of ~1.0 perform best according to this scoring metric. In general, `scikit` has many tools for finding optimized hyperparameters for a given model - an example is [GridSearchCV](http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html), which automatically goes through all possible combinations of hyperparameters, finding the best one. You should note that there is a problem with using the Cross-validation scores we compute here to evaluate the performance of a given set of hyperparameters, which is very similar to the original problem of overfitting. Can you see what it is?
# 

# In[ ]:





# 
# # Testing and evaluating the model
# 
# Now we've reached the stage where we start actually doing electronic structure calculations!
# 
# We would like to test the models we have made by comparing the predicted quantities to calculated quantities. We are looking for a material that is not present in the database, and fulfills the following criteria:
# * It has a negative heat of formation
# * The cubic symmetry is the most stable for this composition class (due to computational cost of the other symmetry types)
# * It has a band gap of approximately 1.5 eV
# 
# It might not be possible to find a material that fulfills all, so try to find one, where the band gap of the cubic symmetry structure is close to 1.5 eV.
# 

# In[ ]:


# Determine all the materials in the database, that do not have the cubic symmetry calculated
names = set(r.name for r in db.select("symmetry!=cubic", project="organometal")) - \
        set(r.name for r in db.select(project="organometal", symmetry="cubic"))



# 
# Ideally there is a structure similar to it in the database. To generate the new structure, we can therefore start with the atoms object of the similar structure. Suppose the identical structure is already present, only with lead instead of tin in the structure. We can then generate an initial guess for the starting structure.
# 

# In[ ]:


row = next(db.select(name='similar name', symmetry='similar symmetry'))
atoms = row.toatoms()
symbols = atoms.get_chemical_symbols()
new_symbols = [? if symbol == ? else symbol
               for symbol in symbols]
atoms.set_chemical_symbols(new_symbols)
view(atoms)
atoms.write("chosen_material.xyz")


# 
# Hopefully, you should see that the structure looks as expected! Unfortunately, the new structure is not in a relaxed configuration - by changing some of the atoms, we've also changed the forces on each atom. Before we calculate the energies and band gaps, we need to relax the structure. Note: because of the computational cost, select a cubic structure.
# 
# **Tip**: You can save the following cell to a file, say `myrelax.py`, by uncommenting the first line and using the next cell to submit it to the cluster by writing the following in a cell:
# ~~~
# !mq submit myrelax.py -R 8:1:4h
# ~~~
# 

# In[ ]:


#%%writefile myrelax.py
from gpaw import GPAW, FermiDirac, PW
from ase.io import read
from ase.optimize.bfgs import BFGS
from ase.filters import UnitCellFilter

atoms = read('chosen_material.xyz')
name = atoms.get_chemical_formula()
calc = GPAW(mode=PW(500),
            kpts={'size': (4, 4, 4), 'gamma': True},
            xc='PBE',
            txt=name + '_relax.out',
            occupations=FermiDirac(width=0.05))

atoms.calc = calc
uf = UnitCellFilter(atoms, mask=[1, 1, 1, 0, 0, 0])
relax = BFGS(uf, logfile=name + '_relax.log', trajectory='chosen_relax.traj')
relax.run(fmax=0.05)  # force is really a stress here


# 
# Once we have the relaxed structure, we are ready to roll! We need to calculate the heat of formation and band gap of this structure, and compare with our predicted values. Time permitting, we should also calculate the heat of formation of the three competing crystal symmetries, to really confirm that we are looking at the correct state. Standard DFT seriously underestimates the band gap. We thus use a more accurate method which includes the calculation of the derivative discontinuity, called GLLBSC. You can find more information about it [here](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.82.115106) and a benchmark of different methodologies [here](https://doi.org/10.1002/aenm.201400915).
# 

# In[ ]:


#%%writefile dft-gllb.py
from ase.io import read
from gpaw import GPAW, FermiDirac, PW

atoms = read('chosen_relax.traj')
calc = GPAW(mode=PW(500),
            kpts={'size': (8, 8, 8), 'gamma': True},
            xc='GLLBSC',
            occupations=FermiDirac(width=0.05))

atoms.calc = calc
energy = atoms.get_potential_energy()

# Note! An accurate discontinuity calculation requires a k-point grid that
# gives accurate HOMO/VBM and LUMO/CBM levels (in other words, the k-points of
# the valence band maximum and the conduction band minimum should be
# included in the used k-point grid).
homo, lumo = calc.get_homo_lumo()
response = calc.hamiltonian.xc.response
dxc_pot = response.calculate_discontinuity_potential(homo, lumo)
KS_gap, dxc = response.calculate_discontinuity(dxc_pot)
gap = KS_gap + dxc
print(f"The gap is {gap:.3f} with components: Kohn-Sham gap {KS_gap:.3f} and "
      f"discontinuity gap of {dxc:.3f}.")


# 
# Do the resulting energies match your predictions?
# 
