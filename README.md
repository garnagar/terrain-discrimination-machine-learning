# Terrain Discrimination Using Machine Learning
Final project for Machine Learning class at School of Engineering of University of Valencia.

## Team
* [Lukas Kyzlik](https://github.com/garnagar)
* [Hanna Olbert](https://github.com/hannaol)

## Assignment
Consider the data in the file `terrain.mat`. It contains 400 height maps corresponding to the relief of mountain landscapes of four different classes: hills, rough rocky terrain, mountains+plain and mountains+valleys.

You have a python script that helps you visualize some of the landscapes.

The problem consists in discriminating from just ONE of the four landscape types from the others. Each team will have a particular landscape type assigned (to be done in the classroom) and there will be 2 different classification problems:

a) discriminate your landscape from another one (you choose)

b) discriminate your landscape from all the remaining ones.

To "solve" your problem you can apply anything studied in the course. In all cases you should use only TWO different class labels, corresponding to your landscape type and others.

You can apply clustering on the "others" class to discover groups in this class and try to make profit of this (without using the true class labels!).

You can also use dimensionality reduction either to reduce noise or redundancy or to display information.

At the end you should obtain one or several predictors and you should assess their behavior using appropriate estimates.

The way in which things have to be done and presented is relatively open and you are supposed to take your decisions. Just imagine that a client has given you that data and that you have to convince him/her.
