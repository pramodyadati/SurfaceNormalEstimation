function layers=conv_eigen(convl,layers,n)

    leng=size(convl);
    layers(1,n).w=zeros(leng(3),leng(4),leng(2),leng(1));
    
    for i=1:leng(3)
        for j=1:leng(4)
            for k=1:leng(2)
                layers(1,n).w(i,j,k,:)=convl(:,k,i,j);
            end
        end
    end