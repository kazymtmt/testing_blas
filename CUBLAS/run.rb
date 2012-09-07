Dir.mkdir "dat" unless File.exist? "dat"

stride = 1
error_check = 0

5.times do
  #[0,1].each do |transa|
  [0,1].each do |order|
    [0].each do |transa|
      #[0,1].each do |transb|
      [0].each do |transb|
        ["sgemm","dgemm"].each do |prog|
          #outfile = "dat/cublas4.2.9_#{`hostname`.chop}_#{prog}_"
          outfile = "dat/cublas5.0rc_#{`hostname`.chop}_#{prog}_"
          outfile += (order == 0) ? "C" : "R"
          outfile += (transa == 0) ? "N" : "T"
          outfile += (transb == 0) ? "N" : "T"
          outfile += ".txt"
          puts outfile
          max_size = (prog == "dgemm") ? 6144 : 7936
          system "./#{prog} #{order} #{transa} #{transb} #{max_size} #{stride} #{error_check} | tee -a #{outfile}"
        end
      end
    end
  end
end

