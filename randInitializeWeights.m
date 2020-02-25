function int = randInitializeWeights(InCon, OutCon)

int = zeros(OutCon, 1+ InCon);
epsilon_int = (sqrt(6)/(sqrt(InCon+OutCon)));
int = rand(OutCon, 1+InCon)*2*epsilon_int - epsilon_int;

end