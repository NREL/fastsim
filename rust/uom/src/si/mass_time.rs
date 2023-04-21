//! Mass time (base unit kilogram second, kg · s).

quantity! {
    /// Mass time (base unit kilogram second, kg · s).
    quantity: MassTime; "mass time";
    /// Dimension of mass time, MT (base unit kilogram second, kg · s).
    dimension: ISQ<
        Z0,     // length
        P1,     // mass
        P1,     // time
        Z0,     // electric current
        Z0,     // thermodynamic temperature
        Z0,     // amount of substance
        Z0>;    // luminous intensity
    units {
        @yottagram_second: prefix!(yotta) / prefix!(kilo); "Yg · s", "yottagram second",
            "yottagram seconds";
        @zettagram_second: prefix!(zetta) / prefix!(kilo); "Zg · s", "zettagram second",
            "zettagram seconds";
        @exagram_second: prefix!(exa) / prefix!(kilo); "Eg · s", "exagram second",
            "exagram seconds";
        @petagram_second: prefix!(peta) / prefix!(kilo); "Pg · s", "petagram second",
            "petagram seconds";
        @teragram_second: prefix!(tera) / prefix!(kilo); "Tg · s", "teragram second",
            "teragram seconds";
        @gigagram_second: prefix!(giga) / prefix!(kilo); "Gg · s", "gigagram second",
            "gigagram seconds";
        @megagram_second: prefix!(mega) / prefix!(kilo); "Mg · s", "megagram second",
            "megagram seconds";
        /// Derived unit of mass time.
        @kilogram_second: prefix!(kilo) / prefix!(kilo); "kg · s", "kilogram second",
            "kilogram seconds";
        @hectogram_second: prefix!(hecto) / prefix!(kilo); "hg · s", "hectogram second",
            "hectogram seconds";
        @decagram_second: prefix!(deca) / prefix!(kilo); "dag · s", "decagram second",
            "decagram seconds";
        @gram_second: prefix!(none) / prefix!(kilo); "g · s", "gram second",
            "gram seconds";
        @decigram_second: prefix!(deci) / prefix!(kilo); "dg · s", "decigram second",
            "decigram seconds";
        @centigram_second: prefix!(centi) / prefix!(kilo); "cg · s", "centigram second",
            "centigram seconds";
        @milligram_second: prefix!(milli) / prefix!(kilo); "mg · s", "milligram second",
            "milligram seconds";
        @microgram_second: prefix!(micro) / prefix!(kilo); "µg · s", "microgram second",
            "microgram seconds";
        @nanogram_second: prefix!(nano) / prefix!(kilo); "ng · s", "nanogram second",
            "nanogram seconds";
        @picogram_second: prefix!(pico) / prefix!(kilo); "pg · s", "picogram second",
            "picogram seconds";
        @femtogram_second: prefix!(femto) / prefix!(kilo); "fg · s", "femtogram second",
            "femtogram seconds";
        @attogram_second: prefix!(atto) / prefix!(kilo); "ag · s", "attogram second",
            "attogram seconds";
        @zeptogram_second: prefix!(zepto) / prefix!(kilo); "zg · s", "zeptogram second",
            "zeptogram seconds";
        @yoctogram_second: prefix!(yocto) / prefix!(kilo); "yg · s", "yoctogram second",
            "yoctogram seconds";

        @kilogram_minute: 6.0_E1; "kg · min", "kilogram minute", "kilogram minutes";
        @kilogram_hour: 3.6_E3; "kg · h", "kilogram hour", "kilogram hours";
        @kilogram_day: 8.64_E4; "kg · d", "kilogram day", "kilogram days";
        @gram_minute: 6.0_E-2; "g · min", "gram minute", "gram minutes";
        @gram_hour: 3.6_E0; "g · h", "gram hour", "gram hours";
        @gram_day: 8.64_E1; "g · d", "gram day", "gram days";

        @carat_second: 2.0_E-4; "ct · s", "carat second", "carat seconds";
        @grain_second: 6.479_891_E-5; "gr · s", "grain second", "grain seconds";
        @hundredweight_long_second: 5.080_235_E1; "cwt long · s", "hundredweight (long) second",
            "hundredweight (long) seconds";
        @hundredweight_short_second: 4.535_924_E1; "cwt short · s",
            "hundredweight (short) second", "hundredweight (short) seconds";
        @ounce_second: 2.834_952_E-2; "oz · s", "ounce second", "ounce seconds";
        @ounce_troy_second: 3.110_348_E-2; "oz t · s", "troy ounce second", "troy ounce seconds";
        @pennyweight_second: 1.555_174_E-3; "dwt · s", "pennyweight second",
            "pennyweight seconds";
        @pound_second: 4.535_924_E-1; "lb · s", "pound second", "pound seconds";
        @pound_minute: 2.721_554_4_E1; "lb · min", "pound minute", "pound minutes";
        @pound_hour: 1.632_932_64_E3; "lb · h", "pound hour", "pound hours";
        @pound_day: 3.919_038_336_E4; "lb · d", "pound day", "pound days";
        @pound_troy_second: 3.732_417_E-1; "lb t · s", "troy pound second", "troy pound seconds";
        @slug_second: 1.459_390_E1; "slug · s", "slug second", "slug seconds";
        @ton_assay_second: 2.916_667_E-2; "AT · s", "assay ton second", "assay ton seconds";
        @ton_long_second: 1.016_047_E3; "2240 lb · s", "long ton second", "long ton seconds";
        @ton_short_second: 9.071_847_E2; "2000 lb · s", "short ton second", "short ton seconds";
        @ton_short_minute: 5.443_108_2_E4; "2000 lb · m", "short ton minute", "short ton minutes";
        @ton_short_hour: 3.265_864_92_E6; "2000 lb · h", "short ton hour", "short ton hours";
        @ton_short_day: 7.838_075_808_E7; "2000 lb · d", "short ton day", "short ton days";
        @ton_second: 1.0_E3; "t · s", "ton second", "ton seconds"; // ton second, metric
    }
}

#[cfg(test)]
mod test {
    storage_types! {
        use crate::num::One;
        use crate::si::mass as m;
        use crate::si::mass_time as c;
        use crate::si::quantities::*;
        use crate::si::time as t;
        use crate::tests::Test;

        #[test]
        fn check_dimension() {
            let _: MassTime<V> = Mass::new::<m::kilogram>(V::one())
                * Time::new::<t::second>(V::one());
        }

        #[test]
        fn check_units() {
            test::<m::yottagram, t::second, c::yottagram_second>();
            test::<m::zettagram, t::second, c::zettagram_second>();
            test::<m::exagram, t::second, c::exagram_second>();
            test::<m::petagram, t::second, c::petagram_second>();
            test::<m::teragram, t::second, c::teragram_second>();
            test::<m::gigagram, t::second, c::gigagram_second>();
            test::<m::megagram, t::second, c::megagram_second>();
            test::<m::kilogram, t::second, c::kilogram_second>();
            test::<m::hectogram, t::second, c::hectogram_second>();
            test::<m::decagram, t::second, c::decagram_second>();
            test::<m::gram, t::second, c::gram_second>();
            test::<m::decigram, t::second, c::decigram_second>();
            test::<m::centigram, t::second, c::centigram_second>();
            test::<m::milligram, t::second, c::milligram_second>();
            test::<m::microgram, t::second, c::microgram_second>();
            test::<m::nanogram, t::second, c::nanogram_second>();
            test::<m::picogram, t::second, c::picogram_second>();
            test::<m::femtogram, t::second, c::femtogram_second>();
            test::<m::attogram, t::second, c::attogram_second>();
            test::<m::zeptogram, t::second, c::zeptogram_second>();
            test::<m::yoctogram, t::second, c::yoctogram_second>();

            test::<m::kilogram, t::minute, c::kilogram_minute>();
            test::<m::kilogram, t::hour, c::kilogram_hour>();
            test::<m::kilogram, t::day, c::kilogram_day>();
            test::<m::gram, t::minute, c::gram_minute>();
            test::<m::gram, t::hour, c::gram_hour>();
            test::<m::gram, t::day, c::gram_day>();

            test::<m::carat, t::second, c::carat_second>();
            test::<m::grain, t::second, c::grain_second>();
            test::<m::hundredweight_long, t::second, c::hundredweight_long_second>();
            test::<m::hundredweight_short, t::second, c::hundredweight_short_second>();
            test::<m::ounce, t::second, c::ounce_second>();
            test::<m::ounce_troy, t::second, c::ounce_troy_second>();
            test::<m::pennyweight, t::second, c::pennyweight_second>();
            test::<m::pound, t::second, c::pound_second>();
            test::<m::pound, t::minute, c::pound_minute>();
            test::<m::pound, t::hour, c::pound_hour>();
            test::<m::pound, t::day, c::pound_day>();
            test::<m::pound_troy, t::second, c::pound_troy_second>();
            test::<m::slug, t::second, c::slug_second>();
            test::<m::ton_assay, t::second, c::ton_assay_second>();
            test::<m::ton_long, t::second, c::ton_long_second>();
            test::<m::ton_short, t::second, c::ton_short_second>();
            test::<m::ton_short, t::minute, c::ton_short_minute>();
            test::<m::ton_short, t::hour, c::ton_short_hour>();
            test::<m::ton_short, t::day, c::ton_short_day>();
            test::<m::ton, t::second, c::ton_second>();

            fn test<M: m::Conversion<V>, T: t::Conversion<V>, C: c::Conversion<V>>() {
                Test::assert_approx_eq(&MassTime::new::<C>(V::one()),
                    &(Mass::new::<M>(V::one()) * Time::new::<T>(V::one())));
            }
        }
    }
}
