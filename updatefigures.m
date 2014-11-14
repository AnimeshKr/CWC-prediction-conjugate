function updatefigures(fhandle,plot_ye,i,numepochs)
    
    if i > 1 %dont plot first point, its only a point   

        M            = {'Training','Validation'};
        plot_x = [(1:i)', (1:i)']; 

    %   plotting
        figure(fhandle);   

        p = plot(plot_x,plot_ye);
        xlabel('Number of epochs'); ylabel('Error');title('Error');
        legend(p, M,'Location','NorthEast');
        set(gca, 'Xlim',[0,numepochs + 1])

        drawnow;
    end
end
