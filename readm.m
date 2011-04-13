function [z,m] = readm(f, nfiles)
  name=sprintf("%s_%i.dat", f, 0);
  mt = load(name);
  m=zeros(nfiles, size(mt,1));
  z = zeros(nfiles, size(mt,1));
  for i = 1:1:nfiles
    name=sprintf("%s_%i.dat", f, i);
    mt = load(name);
    m(i,:)=mt(:,2);
    z(i,:)=mt(:,1);
  endfor
endfunction
