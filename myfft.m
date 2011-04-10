function [vfft,f] = myfft(v)
  t=v(:,1);
  v=v(:,2);
  dt = t(2)-t(1);
  L=size(t,1);
  N=2^nextpow2(L);
  tfft = fft(v,N);
  df = 1/dt;
  f = df/2*linspace(0,1,N/2+1);
  vfft = abs(tfft(1:N/2+1))/max(abs(tfft(1:N/2+1)));
endfunction
