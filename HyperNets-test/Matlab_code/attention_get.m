function [Output, attention_weight_out] = attention_get(Input,weight1,weight2,d_k_size)
     Q = Input*weight1;
     K = Input*weight2;
     
     S = (Q*K.')./sqrt(d_k_size);
%      for ii = 1:101
%          attention_weight(ii,:) = exp(S(ii,:))./sum(exp(S(ii,:)))
%      end
     attention_weight = softmax(S);
     Output = attention_weight*Input;
     attention_weight_out = diag(attention_weight);
end