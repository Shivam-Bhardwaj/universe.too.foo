//! RTP packetization for H.264 video streams
//!
//! Converts H.264 NAL units into RTP packets for WebRTC transport.

use bytes::{BufMut, Bytes, BytesMut};

/// RTP header size in bytes
const RTP_HEADER_SIZE: usize = 12;

/// Maximum RTP payload size (MTU - IP header - UDP header - RTP header)
const MAX_RTP_PAYLOAD: usize = 1200;

/// H.264 NAL unit types
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum NalUnitType {
    /// Coded slice of a non-IDR picture
    Slice = 1,
    /// Coded slice of an IDR picture
    Idr = 5,
    /// Supplemental enhancement information
    Sei = 6,
    /// Sequence parameter set
    Sps = 7,
    /// Picture parameter set
    Pps = 8,
    /// Access unit delimiter
    Aud = 9,
    /// Single-time aggregation packet A
    StapA = 24,
    /// Fragmentation unit A
    FuA = 28,
    /// Unknown
    Unknown = 0,
}

impl From<u8> for NalUnitType {
    fn from(val: u8) -> Self {
        match val & 0x1F {
            1 => NalUnitType::Slice,
            5 => NalUnitType::Idr,
            6 => NalUnitType::Sei,
            7 => NalUnitType::Sps,
            8 => NalUnitType::Pps,
            9 => NalUnitType::Aud,
            24 => NalUnitType::StapA,
            28 => NalUnitType::FuA,
            _ => NalUnitType::Unknown,
        }
    }
}

/// RTP packetizer for H.264
pub struct H264RtpPacketizer {
    /// Current sequence number
    sequence_number: u16,
    /// SSRC identifier
    ssrc: u32,
    /// Payload type
    payload_type: u8,
    /// Clock rate (90000 for video)
    clock_rate: u32,
}

impl H264RtpPacketizer {
    /// Create a new packetizer
    pub fn new(ssrc: u32, payload_type: u8) -> Self {
        Self {
            sequence_number: rand::random(),
            ssrc,
            payload_type,
            clock_rate: 90000,
        }
    }

    /// Packetize an H.264 access unit (frame)
    ///
    /// Returns a list of RTP packets.
    pub fn packetize(&mut self, nal_units: &[Bytes], timestamp: u32, marker: bool) -> Vec<Bytes> {
        let mut packets = Vec::new();

        for (i, nal_unit) in nal_units.iter().enumerate() {
            let is_last = i == nal_units.len() - 1;
            let marker_bit = marker && is_last;

            if nal_unit.len() <= MAX_RTP_PAYLOAD {
                // Single NAL unit packet
                let packet = self.create_single_nal_packet(nal_unit, timestamp, marker_bit);
                packets.push(packet);
            } else {
                // Fragmentation Unit A (FU-A)
                let mut fragments = self.create_fragmented_packets(nal_unit, timestamp, marker_bit);
                packets.append(&mut fragments);
            }
        }

        packets
    }

    /// Create a single NAL unit RTP packet
    fn create_single_nal_packet(&mut self, nal_unit: &[u8], timestamp: u32, marker: bool) -> Bytes {
        let mut buf = BytesMut::with_capacity(RTP_HEADER_SIZE + nal_unit.len());

        // RTP header
        self.write_rtp_header(&mut buf, timestamp, marker);

        // NAL unit payload
        buf.put_slice(nal_unit);

        buf.freeze()
    }

    /// Create Fragmentation Unit A packets for large NAL units
    fn create_fragmented_packets(&mut self, nal_unit: &[u8], timestamp: u32, marker: bool) -> Vec<Bytes> {
        let mut packets = Vec::new();

        // Extract NAL header
        let nal_header = nal_unit[0];
        let nal_type = nal_header & 0x1F;
        let nri = nal_header & 0x60;

        // FU-A header
        let fu_indicator = (nri | NalUnitType::FuA as u8) as u8;

        // Fragment the NAL unit (skip first byte which is the header)
        let payload = &nal_unit[1..];
        let max_fragment_size = MAX_RTP_PAYLOAD - 2; // -2 for FU indicator and header

        let mut offset = 0;
        let mut is_first = true;

        while offset < payload.len() {
            let remaining = payload.len() - offset;
            let fragment_size = remaining.min(max_fragment_size);
            let is_last = offset + fragment_size >= payload.len();

            // FU header
            let mut fu_header = nal_type;
            if is_first {
                fu_header |= 0x80; // Start bit
            }
            if is_last {
                fu_header |= 0x40; // End bit
            }

            let mut buf = BytesMut::with_capacity(RTP_HEADER_SIZE + 2 + fragment_size);

            // RTP header
            let marker_bit = marker && is_last;
            self.write_rtp_header(&mut buf, timestamp, marker_bit);

            // FU-A header
            buf.put_u8(fu_indicator);
            buf.put_u8(fu_header);

            // Fragment payload
            buf.put_slice(&payload[offset..offset + fragment_size]);

            packets.push(buf.freeze());

            offset += fragment_size;
            is_first = false;
        }

        packets
    }

    /// Write RTP header to buffer
    fn write_rtp_header(&mut self, buf: &mut BytesMut, timestamp: u32, marker: bool) {
        // Version (2), padding (0), extension (0), CSRC count (0)
        buf.put_u8(0x80);

        // Marker + payload type
        let byte2 = if marker {
            0x80 | self.payload_type
        } else {
            self.payload_type
        };
        buf.put_u8(byte2);

        // Sequence number
        buf.put_u16(self.sequence_number);
        self.sequence_number = self.sequence_number.wrapping_add(1);

        // Timestamp
        buf.put_u32(timestamp);

        // SSRC
        buf.put_u32(self.ssrc);
    }

    /// Parse H.264 Annex B byte stream into NAL units
    pub fn parse_annex_b(data: &[u8]) -> Vec<Bytes> {
        let mut nal_units = Vec::new();
        let mut start = 0;
        let mut i = 0;

        while i < data.len() {
            // Look for start code (0x00 0x00 0x01 or 0x00 0x00 0x00 0x01)
            if i + 3 <= data.len() && data[i] == 0 && data[i + 1] == 0 {
                let (start_code_len, found) = if data[i + 2] == 1 {
                    (3, true)
                } else if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                    (4, true)
                } else {
                    (0, false)
                };

                if found {
                    // Found start code
                    if start < i {
                        // Save previous NAL unit
                        nal_units.push(Bytes::copy_from_slice(&data[start..i]));
                    }
                    start = i + start_code_len;
                    i = start;
                    continue;
                }
            }
            i += 1;
        }

        // Save last NAL unit
        if start < data.len() {
            nal_units.push(Bytes::copy_from_slice(&data[start..]));
        }

        nal_units
    }

    /// Calculate RTP timestamp from frame number and frame rate
    pub fn calculate_timestamp(&self, frame_number: u64, fps: u32) -> u32 {
        let ticks_per_frame = self.clock_rate / fps;
        ((frame_number * ticks_per_frame as u64) % (u32::MAX as u64 + 1)) as u32
    }
}

impl Default for H264RtpPacketizer {
    fn default() -> Self {
        Self::new(rand::random(), 96)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_annex_b() {
        // Test data with two NAL units
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1E, // SPS with 4-byte start code
            0x00, 0x00, 0x01, 0x68, 0xCE, 0x38, 0x80,       // PPS with 3-byte start code
        ];

        let nal_units = H264RtpPacketizer::parse_annex_b(&data);

        assert_eq!(nal_units.len(), 2);
        assert_eq!(nal_units[0][0] & 0x1F, 7); // SPS
        assert_eq!(nal_units[1][0] & 0x1F, 8); // PPS
    }

    #[test]
    fn test_small_nal_packetization() {
        let mut packetizer = H264RtpPacketizer::new(12345, 96);

        let nal_unit = Bytes::from(vec![0x67, 0x42, 0x00, 0x1E, 0xDA]);
        let packets = packetizer.packetize(&[nal_unit], 0, true);

        assert_eq!(packets.len(), 1);
        // Check RTP header
        assert_eq!(packets[0][0], 0x80); // Version 2
        assert_eq!(packets[0][1], 0x80 | 96); // Marker + PT
    }

    #[test]
    fn test_large_nal_fragmentation() {
        let mut packetizer = H264RtpPacketizer::new(12345, 96);

        // Create a NAL unit larger than MAX_RTP_PAYLOAD
        let large_data: Vec<u8> = (0..3000).map(|i| (i % 256) as u8).collect();
        let mut nal_unit = vec![0x65]; // IDR NAL type
        nal_unit.extend(large_data);

        let packets = packetizer.packetize(&[Bytes::from(nal_unit)], 0, true);

        // Should be fragmented into multiple packets
        assert!(packets.len() > 1);

        // First packet should have FU-A indicator and start bit
        assert_eq!(packets[0][RTP_HEADER_SIZE] & 0x1F, 28); // FU-A
        assert_eq!(packets[0][RTP_HEADER_SIZE + 1] & 0x80, 0x80); // Start bit

        // Last packet should have end bit and marker
        let last = &packets[packets.len() - 1];
        assert_eq!(last[RTP_HEADER_SIZE + 1] & 0x40, 0x40); // End bit
        assert_eq!(last[1] & 0x80, 0x80); // Marker bit
    }
}
