%%
% <latex>
% \title{BE 521 - Homework 3\\{\normalsize Spring 2015}}
% \author{Mike Lautman}
% \date{\today}
% \maketitle
% \textbf{Objective:} Computational modeling of neurons.
% </latex>

close all; clear all; clc; 
%%
% <latex>
% \section*{1. Basic Membrane and Equilibrium Potentials (5 pts)}
% </latex>

time = 3;             % S
wave_f = 2;           % Hz
sample_f = 100;       % Hz

s= 0;              
e = 2 * pi * wave_f * time;     % rad
steps = time * sample_f; 

x = linspace(start, finish, steps);
y = sin(x);

plot(x,y)
title('2 Hz sin Wave'); 
ylabel('Amplitude');
xlabel('time (S)')