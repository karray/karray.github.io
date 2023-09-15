module Jekyll
    class WakaTag < Liquid::Tag
        def initialize(tag_name, input, tokens)
            super

        end
    
        def render(context)
            data = JSON.parse(File.read("_data/waka.json"))['data'].filter{|x| x['percent'] > 1 && x['name'] != 'Other'}

            total_percent = data.map{|x| x['percent']}.sum

            data << {
                'name' => 'Other',
                'percent' => 100 - total_percent
            }
            
            result = '<table id="waka">'
            result += data.map{|x| make_bar(x['name'], x['percent'])}.join('')
            result += '</table>'
            return result;
        end

        def make_bar(title, p, sections=40)
            x = p * sections/100;
            decimal = x.floor
            rest = x - decimal

            result = '█'*decimal
            result += rest > 0.5 ? '▓' : '▒'
            result += '░'*(sections - decimal - 1)
    
            return "<tr><td>#{title}</td><td>#{result}</td><td>#{'%05.2f' % p}%</td></tr>"
        end
    end
end

Liquid::Template.register_tag('waka', Jekyll::WakaTag)