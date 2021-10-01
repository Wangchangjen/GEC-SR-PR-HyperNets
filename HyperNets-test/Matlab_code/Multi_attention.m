function [Output2, Attentions] = Multi_attention(Input,Input_Weights,Multi_num,d_k_size)
    Outputs=[];
    Attentions =[];
    if Multi_num > 1
        for ii=1:Multi_num
            if ii == 1
                weight1 = Input_Weights.array_W1;
                weight2 = Input_Weights.array_W2;
                [Output, Attention] = attention_get(Input,weight1,weight2,d_k_size);
                Attentions = [Attentions,Attention];
                Outputs = [Outputs, Output];
            end
            if ii == 2
                weight1 = Input_Weights.array_W6;
                weight2 = Input_Weights.array_W7;
                [Output, Attention] = attention_get(Input,weight1,weight2,d_k_size);
                Attentions = [Attentions,Attention];
                Outputs = [Outputs, Output];
            end
            
            if ii == 3
                weight1 = Input_Weights.array_W8;
                weight2 = Input_Weights.array_W9;
                [Output, Attention] = attention_get(Input,weight1,weight2,d_k_size);
                Attentions = [Attentions,Attention];
                Outputs = [Outputs, Output];
            end
            
            if ii == 4
                weight1 = Input_Weights.array_W10;
                weight2 = Input_Weights.array_W11;
                [Output, Attention] = attention_get(Input,weight1,weight2,d_k_size);
                Attentions = [Attentions,Attention];
                Outputs = [Outputs, Output];
            end  
                
        end
    else
            weight1 = Input_Weights.array_W1;
            weight2 = Input_Weights.array_W2;
            [Output, Attention] = attention_get(Input,weight1,weight2,d_k_size);
            Attentions = [Attentions,Attention];
            Outputs = [Outputs, Output];
    end
    
    Output2 = (Outputs*Input_Weights.array_W5.');
end