#!/bin/bash
for i in {1..20}; do
    python /home/andre/GitHub/MultiagentPacman/pacman.py --frameTime 0 -p ReflexAgent -k 2 #-g DirectionalGhost
done