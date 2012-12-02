ROUTINE = "dgemm"
STRIDE = 64
MAX_SIZE = 4096
TRIALS = 1

def get_blaslib_name
  open("../Makefile.var", "r") do |r|
    return r.gets.strip.scan(/\/([\w\-\.]+)\z/).to_s
  end
  return "unknown"
end

BLASLIB_NAME = get_blaslib_name()
Dir.mkdir "dat" unless File.exist? "dat"

error_check = 0

[0].each do |order|
  [0].each do |transa|
    [0].each do |transb|
      TRIALS.times do
        outfile = "dat/#{ROUTINE}_" 
        outfile += (order == 0)  ? "C" : "R"
        outfile += (transa == 0) ? "N" : "T"
        outfile += (transb == 0) ? "N" : "T"
        outfile += "_#{`hostname`.strip}_#{BLASLIB_NAME}.txt"
        puts outfile
        system "./testing_#{ROUTINE} #{order} #{transa} #{transb} #{MAX_SIZE} #{STRIDE} #{error_check} | tee -a #{outfile}"
      end
    end
  end
end

