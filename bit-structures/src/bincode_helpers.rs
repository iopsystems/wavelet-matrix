// Helper function for more concise bincode definitions.
// They need to be manually kept up to date with the type
// (todo: can we add serialization to the bitvec trait and test roundtrips?)
// > bincode_encode_impl!(levels, max_symbol, len);
// becomes
// > bincode::Encode::encode(&self.levels, encoder)?;
// > bincode::Encode::encode(&self.max_symbol, encoder)?;
// > bincode::Encode::encode(&self.len, encoder)?;

// fn encode<E: bincode::enc::Encoder>(
//     &self,
//     encoder: &mut E,
// ) -> core::result::Result<(), bincode::error::EncodeError> {
//     bincode::Encode::encode(&self.raw, encoder)?;
//     bincode::Encode::encode(&self.sr_pow2, encoder)?;
//     bincode::Encode::encode(&self.ss_pow2, encoder)?;
//     bincode::Encode::encode(&self.r, encoder)?;
//     bincode::Encode::encode(&self.s0, encoder)?;
//     bincode::Encode::encode(&self.s1, encoder)?;
//     bincode::Encode::encode(&self.num_ones, encoder)?;
//     Ok(())
// }
macro_rules! bincode_encode_impl {
    ($($t:ident),* $(,)?) => (
        fn encode<E: bincode::enc::Encoder>(
            &self,
            encoder: &mut E,
        ) -> core::result::Result<(), bincode::error::EncodeError> {
            $(bincode::Encode::encode(&self.$t, encoder)?;)*
            Ok(())
        }
    )
}

// fn decode<D: bincode::de::Decoder>(
//     decoder: &mut D,
// ) -> core::result::Result<Self, bincode::error::DecodeError> {
//     Ok(Self {
//         raw: bincode::Decode::decode(decoder)?,
//         sr_pow2: bincode::Decode::decode(decoder)?,
//         ss_pow2: bincode::Decode::decode(decoder)?,
//         r: bincode::Decode::decode(decoder)?,
//         s0: bincode::Decode::decode(decoder)?,
//         s1: bincode::Decode::decode(decoder)?,
//         num_ones: bincode::Decode::decode(decoder)?,
//     })
// }
macro_rules! bincode_decode_impl {
    ($($t:ident),* $(,)?) => (
        fn decode<D: bincode::de::Decoder>(
            decoder: &mut D,
        ) -> core::result::Result<Self, bincode::error::DecodeError> {
            Ok(Self {
                $($t: bincode::Decode::decode(decoder)?,)*
            })
        }
    )
}

// fn borrow_decode<D: bincode::de::BorrowDecoder<'de>>(
//     decoder: &mut D,
// ) -> core::result::Result<Self, bincode::error::DecodeError> {
//     Ok(Self {
//         raw: bincode::BorrowDecode::borrow_decode(decoder)?,
//         sr_pow2: bincode::BorrowDecode::borrow_decode(decoder)?,
//         ss_pow2: bincode::BorrowDecode::borrow_decode(decoder)?,
//         r: bincode::BorrowDecode::borrow_decode(decoder)?,
//         s0: bincode::BorrowDecode::borrow_decode(decoder)?,
//         s1: bincode::BorrowDecode::borrow_decode(decoder)?,
//         num_ones: bincode::BorrowDecode::borrow_decode(decoder)?,
//     })
// }
macro_rules! bincode_borrow_decode_impl {
    ($($t:ident),* $(,)?) => (
        fn borrow_decode<D: bincode::de::BorrowDecoder<'de>>(
            decoder: &mut D,
        ) -> core::result::Result<Self, bincode::error::DecodeError> {
            Ok(Self {
                $($t: bincode::BorrowDecode::borrow_decode(decoder)?,)*
            })
        }
    )
}

pub(crate) use bincode_borrow_decode_impl;
pub(crate) use bincode_decode_impl;
pub(crate) use bincode_encode_impl;
