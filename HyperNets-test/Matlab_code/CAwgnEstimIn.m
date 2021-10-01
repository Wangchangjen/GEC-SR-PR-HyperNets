classdef CAwgnEstimIn
    % CAwgnEstimIn:  Circular AWGN scalar input estimation function
    
    properties
        % Prior mean and variance
        mean0 = 0;         % Mean
        var0 = 1;          % Variance
        rho0 = 1;           % Sparse 
    end
    
    
    methods
        % Constructor
        function obj = CAwgnEstimIn(mean0, var0, rho0)
            % obj = obj@EstimIn;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.mean0 = mean0;
                obj.var0 = var0;
                obj.rho0 = rho0;
                
                % warn user about inputs
                if any((var0(:)<0))||any(~isreal(var0(:))),
                    error('Second argument of CAwgnEstimIn must be non-negative');
                end;
            end
        end
        
        %Set Methods
        function obj = set.mean0(obj, mean0)
            obj.mean0 = mean0;
        end
        
        function obj = set.var0(obj, var0)
            assert(all(var0(:) > 0), ...
                'CAwgnEstimIn: variances must be positive');
            obj.var0 = var0;
        end
        
        
        % Circular AWGN estimation function
        % Provides the mean and variance of a variable x = CN(uhat0,uvar0)
        % from an observation rhat = x + w, w = CN(0,rvar)
        function [xhat, xvar] = estim(obj, rhat, rvar)
            % Get prior
            xhat0 = obj.mean0;
            xvar0 = max(eps,obj.var0); % avoid zero variances!
            xrho0 = obj.rho0;
            
            % Compute posterior mean and variance
            
            a = exp(-abs(rhat).^2 ./ rvar + abs(xhat0 - rhat).^2 ./ (xvar0 + rvar) );
            c = 1 ./ (pi .* (rvar + xvar0) );
            Z = (1 - xrho0) ./ (pi .* rvar) .* a + xrho0 .* c;
            
            xhat = (xrho0 .* c .* (xhat0 .* rvar + rhat .* xvar0) ./ (rvar + xvar0) ) ./ Z;
            inv_rvar = 1./rvar ; inv_xvar0 = 1./xvar0 ;
            x2hat = (xrho0 ./ (pi .* rvar .* xvar0) .* (inv_rvar + inv_xvar0).^(-2) .* (1 + abs(xhat0.*inv_xvar0 + rhat.*inv_rvar).^2 ./ (inv_rvar + inv_xvar0) ) ) ./ Z;
            xvar = max(1e-18, x2hat - abs(xhat).^2);
            
            
        end
        
        
        
    end
    
end

