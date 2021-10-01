function [damping, h_out] = GRU_att_damp(Input_x,Input_h,weights,d_k_size)
weight1 = weights.array_W1;
weight2 = weights.array_W2; 
weight3 = weights.array_W3;
weight4 = weights.array_W4; 
weight5 = weights.array_W5;
weight6 = weights.array_W6;

Inputs = [Input_x,Input_h];
r_gate = sigmoid_f(Inputs*weight1);
z_gate = sigmoid_f(Inputs*weight2);
h_r = r_gate .* Input_h;
Inputs2 = [Input_x, h_r];
h_r2 = tanh_f(Inputs2*weight3);
h_out = (1-z_gate).* Input_h + z_gate .* h_r2;

[Output, Attention] = attention_get(h_out.',weight5,weight6,d_k_size);

damping = sigmoid_f(Output.'*weight4);
end
