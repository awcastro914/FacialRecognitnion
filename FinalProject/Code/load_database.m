function out=load_database();
persistent loaded;
persistent w;
if(isempty(loaded))
    v=zeros(10304,400);
    for i=1:40
        for j=1:10
            a=strcat('pics\facePics\orl_faces\s',(num2str(i)));
            a= strcat(a,'\');
            a=imread(strcat (a,(strcat(num2str(j),'.pgm'))));
            v(:,(i-1)*10+j)=reshape(a,size(a,1)*size(a,2),1);
        end
        %cd ..
    end
    w=uint8(v); % 
end
loaded=1;  
out=w;