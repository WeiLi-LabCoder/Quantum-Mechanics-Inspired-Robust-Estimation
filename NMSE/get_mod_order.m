function M = get_mod_order(modStr)
    switch upper(modStr)
        case 'QPSK',  M = 4;
        case '16QAM', M = 16;
        case '64QAM', M = 64;
        otherwise,    M = 16;
    end
end