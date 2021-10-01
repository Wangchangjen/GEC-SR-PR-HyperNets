classdef UnitaryOperator
    % CAwgnEstimIn:  Circular AWGN scalar input estimation function
    
    properties
        % Prior mean and variance
        HaarMatrix = 0;         % Haar Matrix 
        tHaarMatrix = 0;
    end
    
    
    methods
        % Constructor
        function obj = UnitaryOperator(m)
            % obj = obj@NormMatrixOperator; 
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.HaarMatrix = m;
                obj.tHaarMatrix = obj.HaarMatrix' ; 
            else
                error('Please set the permulation of the FFT');
            end
        end
        
        %Set Methods
%         function obj = set.perm(obj, index_perm)
%             obj.perm = index_perm;
%         end 
        
        %  HaarMatrix*xIn
        function [yOut] = Opt(obj, xIn) 
            yOut = obj.HaarMatrix * xIn ; 
        end
        
        % HaarMatrix'*xIn
        function [yOut] = tOpt(obj, xIn)  
            yOut = obj.tHaarMatrix * xIn ; 
        end 
        
         function [yOut] = Opt_adsize(obj, xIn,m) 
            z = obj.HaarMatrix * xIn ; 
            yOut = z(1:m);
         end
         
         function [yOut] = tOpt_adsize(obj, xIn,m) 
            z = obj.tHaarMatrix * xIn ; 
            yOut = z(1:m);
         end
        
    end
    
end

