function Hamil_P=Hamiltonian_Original(N_r,L,const_Hamil)
% This function constructs Hamiltonian matrix without diagonal noisy
% signals



% Construct Hamiltonian
Hamil_P = spalloc(N_r*L, N_r*L, 5*N_r*L); 
types = {'inner', 'edge', 'corner'};

 
% compute the Hamiltonian matrix for vector form
if L ==1
    id_range = [1,3];
    coef_diag = [2,1];
elseif L ==2
    id_range = [2,3];
    coef_diag = [3,2];
else
    id_range = 1:3;
    coef_diag = [4, 3, 2]; % inner, edge, corner
end

for iter = 1:length(id_range)
    type = types{id_range(iter)};
     [X, Y] = selectPoints(type, N_r, L);
    idx = sub2ind([N_r, L], X, Y);
    idx = (Y - 1) * N_r + X;
    adj_idx = getAdjacentIndices(X, Y, N_r, L);
    for kk = 1:length(idx)
        Hamil_P(idx(kk), idx(kk)) = coef_diag(iter)*const_Hamil;
        for neighbor = adj_idx{kk}
            Hamil_P(idx(kk), neighbor) = -const_Hamil;
        end
    end
end
end
 


function [X, Y] = selectPoints(type, N_r, L)
    if L ==1
        switch type
            case 'corner'
                X = [1, N_r];
                Y = [L,L];
            case 'inner'
                X=2:1:N_r-1;
                Y = [L*ones(1,N_r-2)];
        end
    elseif L==2
        switch type
            case 'corner'
                X = [1, N_r, 1, N_r];
                Y = [1, 1, L, L];
            case 'edge'
                X = [2:N_r-1, 2:N_r-1];
                Y = [ones(1,N_r-2), L*ones(1,N_r-2)];
        end
    elseif L>2
        switch type
            case 'corner'
                X = [1, N_r, 1, N_r];
                Y = [1, 1, L, L];
            case 'edge'
                X = [2:N_r-1, ones(1,L-2), N_r*ones(1,L-2), 2:N_r-1];
                Y = [ones(1,N_r-2), 2:L-1, 2:L-1, L*ones(1,N_r-2)];
            case 'inner'
                [X, Y] = meshgrid(2:N_r-1, 2:L-1);
                X = X(:)';
                Y = Y(:)';
        end
    end    
end

function adj_idx = getAdjacentIndices( X, Y, N_r, L)
    adj_idx = cell(1, length(X));
    for kk = 1:length(X)
        ax = [X(kk), X(kk), X(kk)-1, X(kk)+1];
        ay = [Y(kk)-1, Y(kk)+1, Y(kk), Y(kk)];
        valid = ax >= 1 & ax <= N_r & ay >= 1 & ay <= L;
        ax = ax(valid);
        ay = ay(valid);
        adj_idx{kk} = (ay - 1) * N_r + ax;
    end
end
